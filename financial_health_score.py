"""
Financial Health Score Calculator for Indian Micro-cap Stocks

This module calculates a comprehensive financial health score for Indian micro-cap
and penny stocks using weighted metrics:
- Promoter Holding: 30%
- Debt-to-Equity Ratio: 40%
- Free Cash Flow Growth: 30%
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class StockFinancials:
    """Data class to hold stock financial metrics."""
    ticker: str
    promoter_holding: float
    debt_to_equity: float
    fcf_growth: float
    company_name: str = ""


def normalize_promoter_holding(promoter_holding: float) -> float:
    """
    Normalize promoter holding to 0-100 score.
    
    Higher promoter holding indicates stronger alignment with minority shareholders.
    Assumes input is in percentage (0-100).
    
    Args:
        promoter_holding: Promoter holding percentage (0-100)
    
    Returns:
        Normalized score (0-100)
    
    Raises:
        ValueError: If promoter holding is not between 0 and 100
    """
    if not (0 <= promoter_holding <= 100):
        raise ValueError(f"Promoter holding must be between 0 and 100, got {promoter_holding}")
    
    return promoter_holding


def normalize_debt_to_equity(debt_to_equity: float) -> float:
    """
    Normalize debt-to-equity ratio to 0-100 score.
    
    Lower D/E ratio indicates better financial health. Uses inverse scoring:
    - D/E of 0-1.0: Excellent (higher score)
    - D/E of 1.0-2.0: Good
    - D/E of 2.0+: Poor (lower score)
    
    Args:
        debt_to_equity: Debt-to-Equity ratio
    
    Returns:
        Normalized score (0-100)
    
    Raises:
        ValueError: If debt_to_equity is negative
    """
    if debt_to_equity < 0:
        raise ValueError(f"Debt-to-Equity ratio must be non-negative, got {debt_to_equity}")
    
    # Use inverse sigmoid-like function for scoring
    # Optimal D/E is around 0.5-0.8 for healthy micro-caps
    if debt_to_equity <= 0.5:
        score = 100
    elif debt_to_equity <= 1.0:
        score = 100 - ((debt_to_equity - 0.5) / 0.5) * 20  # 100 to 80
    elif debt_to_equity <= 2.0:
        score = 80 - ((debt_to_equity - 1.0) / 1.0) * 50  # 80 to 30
    else:
        score = max(10, 30 - ((debt_to_equity - 2.0) * 5))  # Floors at 10
    
    return np.clip(score, 0, 100)


def normalize_fcf_growth(fcf_growth: float) -> float:
    """
    Normalize Free Cash Flow growth to 0-100 score.
    
    Higher FCF growth indicates better operational efficiency.
    Input can be percentage growth rate.
    
    Args:
        fcf_growth: Free Cash Flow growth rate (in percentage, can be negative)
    
    Returns:
        Normalized score (0-100)
    """
    # FCF growth expectations for micro-caps:
    # Negative or 0%: Poor (score 10)
    # 0-10%: Below average (score 20-40)
    # 10-25%: Good (score 60-80)
    # 25%+: Excellent (score 90-100)
    
    if fcf_growth < 0:
        score = 10
    elif fcf_growth <= 10:
        score = 10 + (fcf_growth / 10) * 30  # 10 to 40
    elif fcf_growth <= 25:
        score = 40 + ((fcf_growth - 10) / 15) * 40  # 40 to 80
    else:
        # Cap at 100 for high growth
        growth_bonus = min((fcf_growth - 25) / 25 * 20, 20)  # Max +20 bonus
        score = 80 + growth_bonus
    
    return np.clip(score, 0, 100)


def calculate_financial_health_score(
    promoter_holding: float,
    debt_to_equity: float,
    fcf_growth: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate the comprehensive financial health score for a stock.
    
    Args:
        promoter_holding: Promoter holding percentage (0-100)
        debt_to_equity: Debt-to-Equity ratio (non-negative)
        fcf_growth: Free Cash Flow growth rate (percentage)
        weights: Dictionary with keys 'promoter', 'debt_to_equity', 'fcf_growth'
                Defaults to 30%, 40%, 30% respectively
    
    Returns:
        Financial health score (0-100)
    
    Raises:
        ValueError: If inputs are invalid
    """
    if weights is None:
        weights = {
            'promoter': 0.30,
            'debt_to_equity': 0.40,
            'fcf_growth': 0.30
        }
    
    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=0.01):
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    # Normalize individual metrics
    promoter_score = normalize_promoter_holding(promoter_holding)
    de_score = normalize_debt_to_equity(debt_to_equity)
    fcf_score = normalize_fcf_growth(fcf_growth)
    
    # Calculate weighted score
    health_score = (
        weights['promoter'] * promoter_score +
        weights['debt_to_equity'] * de_score +
        weights['fcf_growth'] * fcf_score
    )
    
    return np.clip(health_score, 0, 100)


def score_multiple_stocks(df: pd.DataFrame, custom_weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Calculate financial health scores for multiple stocks in a DataFrame.
    
    Args:
        df: DataFrame with columns: 'ticker', 'promoter_holding', 'debt_to_equity', 'fcf_growth'
        custom_weights: Optional custom weights for metrics
    
    Returns:
        DataFrame with added columns:
            - 'promoter_score': Normalized promoter holding score
            - 'de_score': Normalized D/E ratio score
            - 'fcf_score': Normalized FCF growth score
            - 'financial_health_score': Overall weighted score
            - 'health_rating': Categorical rating (Excellent/Good/Fair/Poor/Critical)
    
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['ticker', 'promoter_holding', 'debt_to_equity', 'fcf_growth']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result_df = df.copy()
    
    # Calculate individual normalized scores
    result_df['promoter_score'] = result_df['promoter_holding'].apply(normalize_promoter_holding)
    result_df['de_score'] = result_df['debt_to_equity'].apply(normalize_debt_to_equity)
    result_df['fcf_score'] = result_df['fcf_growth'].apply(normalize_fcf_growth)
    
    # Calculate overall financial health score
    result_df['financial_health_score'] = result_df.apply(
        lambda row: calculate_financial_health_score(
            row['promoter_holding'],
            row['debt_to_equity'],
            row['fcf_growth'],
            weights=custom_weights
        ),
        axis=1
    )
    
    # Add health rating based on score
    def get_health_rating(score: float) -> str:
        if score >= 80:
            return 'Excellent'
        elif score >= 65:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        elif score >= 35:
            return 'Poor'
        else:
            return 'Critical'
    
    result_df['health_rating'] = result_df['financial_health_score'].apply(get_health_rating)
    
    return result_df


def generate_score_report(df: pd.DataFrame) -> str:
    """
    Generate a formatted report of financial health scores.
    
    Args:
        df: DataFrame with financial health scores (output from score_multiple_stocks)
    
    Returns:
        Formatted string report
    """
    if 'financial_health_score' not in df.columns:
        raise ValueError("DataFrame must contain 'financial_health_score' column")
    
    report = []
    report.append("=" * 80)
    report.append("FINANCIAL HEALTH SCORE REPORT - INDIAN MICRO-CAP STOCKS")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    if len(df) > 0:
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Stocks Analyzed: {len(df)}")
        report.append(f"Average Health Score: {df['financial_health_score'].mean():.2f}")
        report.append(f"Median Health Score: {df['financial_health_score'].median():.2f}")
        report.append(f"Highest Score: {df['financial_health_score'].max():.2f} ({df.loc[df['financial_health_score'].idxmax(), 'ticker']})")
        report.append(f"Lowest Score: {df['financial_health_score'].min():.2f} ({df.loc[df['financial_health_score'].idxmin(), 'ticker']})")
        report.append("")
        
        # Rating distribution
        report.append("RATING DISTRIBUTION")
        report.append("-" * 80)
        rating_counts = df['health_rating'].value_counts().sort_index(ascending=False)
        for rating, count in rating_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"{rating:12} : {count:3} stocks ({percentage:5.1f}%)")
        report.append("")
        
        # Top performers
        report.append("TOP 5 STOCKS BY HEALTH SCORE")
        report.append("-" * 80)
        top_5 = df.nlargest(5, 'financial_health_score')[['ticker', 'financial_health_score', 'health_rating', 'promoter_holding', 'debt_to_equity', 'fcf_growth']]
        for idx, row in top_5.iterrows():
            report.append(f"{row['ticker']:10} | Score: {row['financial_health_score']:6.2f} | "
                        f"Rating: {row['health_rating']:10} | "
                        f"Promoter: {row['promoter_holding']:6.1f}% | "
                        f"D/E: {row['debt_to_equity']:6.2f} | "
                        f"FCF Growth: {row['fcf_growth']:6.1f}%")
        report.append("")
        
        # Bottom performers
        report.append("BOTTOM 5 STOCKS BY HEALTH SCORE")
        report.append("-" * 80)
        bottom_5 = df.nsmallest(5, 'financial_health_score')[['ticker', 'financial_health_score', 'health_rating', 'promoter_holding', 'debt_to_equity', 'fcf_growth']]
        for idx, row in bottom_5.iterrows():
            report.append(f"{row['ticker']:10} | Score: {row['financial_health_score']:6.2f} | "
                        f"Rating: {row['health_rating']:10} | "
                        f"Promoter: {row['promoter_holding']:6.1f}% | "
                        f"D/E: {row['debt_to_equity']:6.2f} | "
                        f"FCF Growth: {row['fcf_growth']:6.1f}%")
    
    report.append("")
    report.append("=" * 80)
    report.append("SCORING METHODOLOGY")
    report.append("=" * 80)
    report.append("Promoter Holding (30%):    Higher percentage = Higher score (0-100)")
    report.append("Debt-to-Equity Ratio (40%): Lower ratio = Higher score (inverse scoring)")
    report.append("FCF Growth Rate (30%):     Higher growth = Higher score (0-100)")
    report.append("=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ticker': ['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5'],
        'promoter_holding': [65.5, 45.2, 72.1, 38.9, 55.3],
        'debt_to_equity': [0.8, 1.5, 0.5, 2.2, 1.1],
        'fcf_growth': [15.3, 8.5, 22.1, -5.2, 18.7]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original Data:")
    print(df)
    print("\n")
    
    # Calculate scores
    scored_df = score_multiple_stocks(df)
    print("Scores Calculated:")
    print(scored_df[['ticker', 'promoter_score', 'de_score', 'fcf_score', 'financial_health_score', 'health_rating']])
    print("\n")
    
    # Generate report
    print(generate_score_report(scored_df))
