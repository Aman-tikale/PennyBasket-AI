"""
Micro-Cap Stock Sentiment Analysis Agent

This module provides AI-powered sentiment analysis for Indian penny stocks
using FinBERT (financial-domain BERT) and VADER sentiment analysis.

The MicroCapSentimentAgent processes financial news headlines and generates
normalized sentiment scores that can be integrated into trading strategies.

Dependencies:
    - transformers>=4.30.0
    - torch>=2.0.0
    - pandas>=1.5.0
    - nltk>=3.8
    - numpy>=1.23.0
"""

import logging
import warnings
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import re

import pandas as pd
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers library not installed. Install with: pip install transformers torch")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    warnings.warn("NLTK library not installed. Install with: pip install nltk")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Data class to hold sentiment analysis results."""
    headline: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    model_used: str
    timestamp: datetime


class MicroCapSentimentAgent:
    """
    AI-powered sentiment analysis agent for Indian micro-cap and penny stocks.
    
    This class utilizes FinBERT (a financial domain-specific BERT model) combined
    with VADER sentiment analysis to provide robust sentiment scoring for news
    headlines and financial texts.
    
    Attributes:
        model_name (str): HuggingFace model identifier
        use_finbert (bool): Whether to use FinBERT (default) or VADER
        device (str): Device to run model on ('cpu' or 'cuda')
        finbert_pipeline (transformers.Pipeline): FinBERT sentiment pipeline
        vader_analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer
    
    Example:
        >>> agent = MicroCapSentimentAgent(use_finbert=True)
        >>> score = agent.analyze_headline_sentiment("Stock XYZ surges on positive earnings")
        >>> print(score)  # Output: 0.85
    """
    
    # Financial sentiment keywords for filtering
    FINANCIAL_KEYWORDS = {
        'positive': [
            'surge', 'rally', 'gain', 'profit', 'growth', 'upside', 'bullish',
            'outperform', 'upgrade', 'beat', 'positive', 'strong', 'excellent',
            'boost', 'rise', 'momentum', 'success', 'record', 'expansion'
        ],
        'negative': [
            'crash', 'plunge', 'loss', 'decline', 'downside', 'bearish',
            'underperform', 'downgrade', 'miss', 'negative', 'weak', 'poor',
            'risk', 'fall', 'selloff', 'concern', 'crisis', 'bankruptcy'
        ],
        'neutral': [
            'report', 'announce', 'trade', 'volume', 'session', 'market',
            'index', 'close', 'open', 'trading', 'data', 'release'
        ]
    }
    
    def __init__(
        self,
        model_name: str = 'yiyanghkust/finbert-tone',
        use_finbert: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the Micro-Cap Sentiment Agent.
        
        Args:
            model_name: HuggingFace model identifier for FinBERT
            use_finbert: If True, use FinBERT; if False, use VADER
            device: Device for model inference ('cpu' or 'cuda')
        
        Raises:
            ImportError: If required libraries are not installed
        """
        self.model_name = model_name
        self.use_finbert = use_finbert
        self.device = device
        self.finbert_pipeline = None
        self.vader_analyzer = None
        
        logger.info(f"Initializing MicroCapSentimentAgent with {model_name}")
        
        # Initialize FinBERT if requested
        if use_finbert:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available. Falling back to VADER.")
                self.use_finbert = False
            else:
                try:
                    logger.info(f"Loading FinBERT model: {model_name}")
                    self.finbert_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device=0 if device == 'cuda' else -1
                    )
                    logger.info("FinBERT model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load FinBERT model: {e}")
                    logger.info("Falling back to VADER sentiment analysis")
                    self.use_finbert = False
        
        # Initialize VADER as fallback or primary
        if not self.use_finbert:
            if not VADER_AVAILABLE:
                logger.error("Neither FinBERT nor VADER available. Install required libraries.")
                raise ImportError("Install transformers or nltk for sentiment analysis")
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
    
    def analyze_headline_sentiment(self, headline: str) -> float:
        """
        Analyze sentiment of a single headline and return normalized score.
        
        This method processes a financial news headline and returns a continuous
        sentiment score normalized between -1 (highly negative) and +1 (highly positive).
        
        Args:
            headline: News headline text to analyze
        
        Returns:
            Sentiment score between -1.0 (negative) and +1.0 (positive)
        
        Raises:
            ValueError: If headline is empty or None
            
        Example:
            >>> agent = MicroCapSentimentAgent()
            >>> score = agent.analyze_headline_sentiment(
            ...     "Microtech Ltd reports record quarterly profits"
            ... )
            >>> print(f"Sentiment: {score:.2f}")  # Output: Sentiment: 0.87
        """
        # Input validation
        if not headline or not isinstance(headline, str):
            logger.warning(f"Invalid headline input: {headline}")
            raise ValueError("Headline must be a non-empty string")
        
        # Clean headline
        headline = headline.strip()
        if len(headline) == 0:
            raise ValueError("Headline cannot be empty after stripping whitespace")
        
        logger.debug(f"Analyzing sentiment for: {headline[:100]}")
        
        try:
            if self.use_finbert and self.finbert_pipeline:
                return self._analyze_with_finbert(headline)
            else:
                return self._analyze_with_vader(headline)
        except Exception as e:
            logger.error(f"Error analyzing headline: {e}")
            raise
    
    def _analyze_with_finbert(self, headline: str) -> float:
        """
        Analyze sentiment using FinBERT model.
        
        Args:
            headline: Text to analyze
        
        Returns:
            Normalized sentiment score (-1 to +1)
        """
        try:
            # Truncate if too long (BERT has 512 token limit)
            max_length = 512
            if len(headline) > max_length * 4:  # Rough approximation
                headline = headline[:max_length * 4]
            
            result = self.finbert_pipeline(headline)[0]
            label = result['label'].lower()
            score = result['score']
            
            # FinBERT returns: negative, neutral, positive
            if label == 'positive':
                normalized_score = score
            elif label == 'negative':
                normalized_score = -score
            else:  # neutral
                normalized_score = 0.0
            
            logger.debug(f"FinBERT result: {label} ({score:.4f}) -> {normalized_score:.4f}")
            return np.clip(normalized_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            raise
    
    def _analyze_with_vader(self, headline: str) -> float:
        """
        Analyze sentiment using VADER sentiment analyzer.
        
        Args:
            headline: Text to analyze
        
        Returns:
            Normalized sentiment score (-1 to +1)
        """
        try:
            scores = self.vader_analyzer.polarity_scores(headline)
            # VADER returns compound score already in -1 to +1 range
            compound = scores['compound']
            
            logger.debug(f"VADER result: {scores}")
            return compound
            
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            raise
    
    def _is_relevant_headline(self, headline: str) -> bool:
        """
        Filter headlines for financial relevance.
        
        Checks if headline contains financial keywords or stock references.
        
        Args:
            headline: Headline text
        
        Returns:
            True if headline is relevant to stock sentiment, False otherwise
        """
        headline_lower = headline.lower()
        
        # Check for financial keywords
        all_keywords = (
            self.FINANCIAL_KEYWORDS['positive'] +
            self.FINANCIAL_KEYWORDS['negative'] +
            self.FINANCIAL_KEYWORDS['neutral']
        )
        
        has_financial_keyword = any(kw in headline_lower for kw in all_keywords)
        
        # Check for stock/company patterns
        has_stock_reference = bool(re.search(
            r'\b(?:stock|share|equity|price|trading|market|bse|nse|rupees?|₹)\b',
            headline_lower
        ))
        
        # Check for numeric patterns (prices, percentages)
        has_numerics = bool(re.search(r'\d+(?:\.\d+)?%|\₹\d+|Rs\.?\s*\d+', headline))
        
        is_relevant = has_financial_keyword or has_stock_reference or has_numerics
        
        if not is_relevant:
            logger.debug(f"Headline filtered as not relevant: {headline[:60]}")
        
        return is_relevant
    
    def _calculate_volume_weight(self, headline_index: int, total_headlines: int) -> float:
        """
        Calculate volume weight for a headline based on recency.
        
        More recent headlines get higher weight (exponential decay).
        
        Args:
            headline_index: Index of headline (0 = oldest)
            total_headlines: Total number of headlines
        
        Returns:
            Weight between 0 and 1
        """
        if total_headlines <= 1:
            return 1.0
        
        # Exponential decay: recent headlines weighted higher
        # Formula: e^(2 * (index / total - 1))
        decay_factor = np.exp(2.0 * (headline_index / (total_headlines - 1) - 1))
        return decay_factor
    
    def aggregate_stock_sentiment(
        self,
        headlines: List[str],
        filter_relevance: bool = True,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Aggregate sentiment from multiple headlines into a volume-weighted index.
        
        Processes a list of news headlines for a specific Indian penny stock,
        filters out irrelevant content, and calculates a weighted sentiment score.
        
        Args:
            headlines: List of news headlines
            filter_relevance: If True, filter out non-financial headlines
            return_details: If True, return detailed analysis dict; else return float
        
        Returns:
            If return_details=False: Aggregated sentiment score (-1 to +1)
            If return_details=True: Dictionary with:
                - 'aggregate_score': Main sentiment score
                - 'individual_scores': List of individual headline scores
                - 'weights': List of weights applied
                - 'confidence': Confidence measure
                - 'headlines_analyzed': Number of headlines processed
                - 'headlines_filtered': Number of headlines filtered out
        
        Raises:
            ValueError: If headlines list is empty or None
            
        Example:
            >>> headlines = [
            ...     "Stock ABC rallies on strong earnings",
            ...     "Company XYZ faces regulatory challenges",
            ...     "Market sentiment turns positive"
            ... ]
            >>> agent = MicroCapSentimentAgent()
            >>> sentiment = agent.aggregate_stock_sentiment(headlines)
            >>> print(f"Aggregate sentiment: {sentiment:.3f}")  # Output: 0.234
        """
        # Input validation
        if not headlines:
            logger.warning("Empty headlines list provided")
            raise ValueError("Headlines list cannot be empty")
        
        if not isinstance(headlines, list):
            logger.error(f"Headlines must be a list, got {type(headlines)}")
            raise TypeError("Headlines must be a list of strings")
        
        # Filter and clean headlines
        valid_headlines = []
        filtered_count = 0
        
        for headline in headlines:
            if not headline or not isinstance(headline, str):
                logger.debug(f"Skipping invalid headline: {headline}")
                filtered_count += 1
                continue
            
            headline = headline.strip()
            if len(headline) == 0:
                filtered_count += 1
                continue
            
            if filter_relevance and not self._is_relevant_headline(headline):
                filtered_count += 1
                continue
            
            valid_headlines.append(headline)
        
        # Handle case where all headlines are filtered
        if not valid_headlines:
            logger.warning(f"All {len(headlines)} headlines filtered as irrelevant")
            if return_details:
                return {
                    'aggregate_score': 0.0,
                    'individual_scores': [],
                    'weights': [],
                    'confidence': 0.0,
                    'headlines_analyzed': 0,
                    'headlines_filtered': len(headlines),
                    'error': 'No relevant headlines found'
                }
            return 0.0
        
        # Analyze each valid headline
        individual_scores = []
        weights = []
        
        for idx, headline in enumerate(valid_headlines):
            try:
                score = self.analyze_headline_sentiment(headline)
                weight = self._calculate_volume_weight(idx, len(valid_headlines))
                
                individual_scores.append(score)
                weights.append(weight)
                
            except Exception as e:
                logger.warning(f"Failed to analyze headline '{headline[:50]}': {e}")
                filtered_count += 1
                continue
        
        # Calculate weighted sentiment
        if not individual_scores:
            logger.warning("No headlines could be analyzed")
            if return_details:
                return {
                    'aggregate_score': 0.0,
                    'individual_scores': [],
                    'weights': [],
                    'confidence': 0.0,
                    'headlines_analyzed': 0,
                    'headlines_filtered': len(headlines),
                    'error': 'Failed to analyze any headlines'
                }
            return 0.0
        
        # Normalize weights to sum to 1
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()
        
        # Calculate weighted average
        aggregate_score = np.average(individual_scores, weights=weights_array)
        
        # Calculate confidence as variance inverse (lower variance = higher confidence)
        if len(individual_scores) > 1:
            variance = np.var(individual_scores)
            # Confidence: 1 - normalized variance
            confidence = 1.0 - np.clip(variance / 2.0, 0, 1)
        else:
            confidence = 0.5  # Medium confidence for single headline
        
        logger.info(
            f"Aggregated sentiment from {len(valid_headlines)} headlines: "
            f"{aggregate_score:.4f} (confidence: {confidence:.2f})"
        )
        
        if return_details:
            return {
                'aggregate_score': float(np.clip(aggregate_score, -1.0, 1.0)),
                'individual_scores': individual_scores,
                'weights': weights_array.tolist(),
                'confidence': float(confidence),
                'headlines_analyzed': len(valid_headlines),
                'headlines_filtered': filtered_count,
                'mean_score': float(np.mean(individual_scores)),
                'std_dev': float(np.std(individual_scores)),
                'timestamp': datetime.now().isoformat()
            }
        
        return float(np.clip(aggregate_score, -1.0, 1.0))
    
    def analyze_headlines_dataframe(
        self,
        df: pd.DataFrame,
        headline_column: str = 'headline',
        ticker_column: str = 'ticker'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for headlines in a pandas DataFrame.
        
        Processes a DataFrame containing headlines and returns a new DataFrame
        with sentiment scores and analysis results.
        
        Args:
            df: DataFrame with headlines
            headline_column: Name of column containing headlines
            ticker_column: Name of column containing stock tickers (optional)
        
        Returns:
            DataFrame with added columns:
                - 'sentiment_score': Raw sentiment score (-1 to +1)
                - 'sentiment_label': Categorical label (positive/neutral/negative)
                - 'sentiment_confidence': Confidence of the prediction
                - 'is_relevant': Whether headline passed relevance filter
        
        Example:
            >>> df = pd.DataFrame({
            ...     'headline': ['Stock X rises', 'Market news'],
            ...     'date': ['2024-01-01', '2024-01-02']
            ... })
            >>> result_df = agent.analyze_headlines_dataframe(df)
        """
        if headline_column not in df.columns:
            raise ValueError(f"Column '{headline_column}' not found in DataFrame")
        
        logger.info(f"Analyzing {len(df)} headlines from DataFrame")
        
        result_df = df.copy()
        
        sentiment_scores = []
        sentiment_labels = []
        confidence_scores = []
        relevance_flags = []
        
        for idx, headline in enumerate(df[headline_column]):
            try:
                # Check relevance
                is_relevant = self._is_relevant_headline(str(headline))
                relevance_flags.append(is_relevant)
                
                if not is_relevant:
                    sentiment_scores.append(0.0)
                    sentiment_labels.append('neutral')
                    confidence_scores.append(0.0)
                    continue
                
                # Analyze sentiment
                score = self.analyze_headline_sentiment(str(headline))
                sentiment_scores.append(score)
                
                # Determine label
                if score > 0.1:
                    label = 'positive'
                elif score < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                sentiment_labels.append(label)
                
                # Use absolute score as confidence proxy
                confidence = min(abs(score), 1.0)
                confidence_scores.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error analyzing headline at index {idx}: {e}")
                sentiment_scores.append(0.0)
                sentiment_labels.append('neutral')
                confidence_scores.append(0.0)
                relevance_flags.append(False)
        
        result_df['sentiment_score'] = sentiment_scores
        result_df['sentiment_label'] = sentiment_labels
        result_df['sentiment_confidence'] = confidence_scores
        result_df['is_relevant'] = relevance_flags
        
        logger.info(f"Sentiment analysis complete. Positive: {sentiment_labels.count('positive')}, "
                   f"Negative: {sentiment_labels.count('negative')}, "
                   f"Neutral: {sentiment_labels.count('neutral')}")
        
        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Starting MicroCapSentimentAgent demonstration")
    
    # Initialize agent
    try:
        agent = MicroCapSentimentAgent(use_finbert=False)  # Use VADER for quick demo
    except ImportError as e:
        logger.error(f"Failed to initialize agent: {e}")
        exit(1)
    
    # Single headline analysis
    print("\n" + "="*80)
    print("SINGLE HEADLINE ANALYSIS")
    print("="*80)
    
    test_headlines = [
        "Stock ABC surges on strong quarterly earnings report",
        "Company XYZ faces regulatory challenges in India",
        "Market sentiment turns bullish for penny stocks",
        "Trading volume drops significantly",
        "Rupee weakness impacts micro-cap valuations"
    ]
    
    for headline in test_headlines:
        try:
            score = agent.analyze_headline_sentiment(headline)
            print(f"Headline: {headline}")
            print(f"Sentiment: {score:.3f}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Aggregate sentiment analysis
    print("="*80)
    print("AGGREGATE SENTIMENT ANALYSIS")
    print("="*80)
    
    aggregated_headlines = [
        "Microtech Ltd rallies 12% on strong Q3 earnings",
        "Stock XYZ falls on profit-taking",
        "Investors bullish on penny stock sector",
        "Trading volume indicates strong institutional interest",
        "Company faces pricing pressure from competitors"
    ]
    
    try:
        agg_score = agent.aggregate_stock_sentiment(
            aggregated_headlines,
            filter_relevance=True,
            return_details=True
        )
        
        print(f"Aggregated Sentiment Score: {agg_score['aggregate_score']:.3f}")
        print(f"Confidence: {agg_score['confidence']:.2f}")
        print(f"Headlines Analyzed: {agg_score['headlines_analyzed']}")
        print(f"Headlines Filtered: {agg_score['headlines_filtered']}")
        print(f"Mean Score: {agg_score['mean_score']:.3f}")
        print(f"Std Dev: {agg_score['std_dev']:.3f}\n")
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
    
    # DataFrame analysis
    print("="*80)
    print("DATAFRAME ANALYSIS")
    print("="*80)
    
    try:
        test_df = pd.DataFrame({
            'headline': aggregated_headlines,
            'date': pd.date_range('2024-01-01', periods=len(aggregated_headlines))
        })
        
        result_df = agent.analyze_headlines_dataframe(test_df, headline_column='headline')
        print(result_df[['headline', 'sentiment_score', 'sentiment_label', 'sentiment_confidence']])
        
    except Exception as e:
        logger.error(f"DataFrame analysis failed: {e}")
