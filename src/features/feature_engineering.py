"""
Feature engineering module with technical indicators, learnable temporal relevance, and market context.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculates comprehensive technical indicators."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.technical_config = config['features']['technical']
        self.volume_config = config['features']['volume']
        self.volatility_config = config['features']['volatility']
        
    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        for period in periods:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        return df
    
    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        return df
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicators."""
        macd = ta.trend.MACD(df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        bollinger = ta.volatility.BollingerBands(df['Close'], window=period, window_dev=std)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                               window=k_period, smooth_window=d_period)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        return df
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], 
                                                         lbp=period).williams_r()
        return df
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=period).cci()
        return df
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 
                                                  window=period).average_true_range()
        df['atr_ratio'] = df['atr'] / df['Close']
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # On-Balance Volume
        if self.volume_config['obv']:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Volume Weighted Average Price
        if self.volume_config['vwap']:
            df['vwap'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
        
        # Volume moving averages
        for period in self.volume_config['volume_sma']:
            df[f'volume_sma_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        # Price-Volume relationship
        if self.volume_config['price_volume']:
            df['price_volume'] = df['Close'] * df['Volume']
            df['price_volume_ma'] = df['price_volume'].rolling(window=20).mean()
        
        return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Historical volatility
        for period in self.volatility_config['volatility_periods']:
            df[f'volatility_{period}'] = df['Returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(252 * np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2)))
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        logger.info("Calculating technical indicators")
        
        # Ensure required columns exist
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            logger.error("Missing required OHLCV columns")
            return df
        
        # Calculate returns if not present
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        # Moving averages
        df = self.calculate_sma(df, self.technical_config['sma_periods'])
        df = self.calculate_ema(df, self.technical_config['ema_periods'])
        
        # Momentum indicators
        df = self.calculate_rsi(df, self.technical_config['rsi_period'])
        df = self.calculate_macd(df, *self.technical_config['macd_params'])
        df = self.calculate_stochastic(df, self.technical_config['stochastic_k'], 
                                     self.technical_config['stochastic_d'])
        df = self.calculate_williams_r(df, self.technical_config['williams_r'])
        df = self.calculate_cci(df, self.technical_config['cci_period'])
        
        # Trend indicators
        df = self.calculate_adx(df, self.technical_config['adx_period'])
        df = self.calculate_bollinger_bands(df, self.technical_config['bollinger_period'], 
                                          self.technical_config['bollinger_std'])
        
        # Volatility indicators
        df = self.calculate_atr(df, self.volatility_config['atr_period'])
        df = self.calculate_volatility_indicators(df)
        
        # Volume indicators
        df = self.calculate_volume_indicators(df)
        
        # Additional price-based features
        df['price_change_pct'] = df['Returns'] * 100
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['volume_price_trend'] = df['Volume'] * df['Returns']
        
        logger.info(f"Calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']])} technical features")
        
        return df


class MarketContextFeatures:
    """Calculates market context and correlation features."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.correlation_config = config['features']['correlation']
        
    def calculate_market_correlations(self, stock_data: Dict[str, pd.DataFrame], 
                                    symbol: str) -> pd.DataFrame:
        """Calculate correlations with market indices."""
        if symbol not in stock_data:
            return pd.DataFrame()
        
        stock_df = stock_data[symbol].copy()
        market_indices = self.config['data']['market_indices']
        
        for index in market_indices:
            if index in stock_data:
                index_data = stock_data[index]
                
                # Align dates
                common_dates = stock_df.index.intersection(index_data.index)
                if len(common_dates) < self.correlation_config['market_corr_period']:
                    continue
                
                # Calculate rolling correlation
                period = self.correlation_config['market_corr_period']
                stock_returns = stock_df.loc[common_dates, 'Returns']
                index_returns = index_data.loc[common_dates, 'Returns']
                
                correlation = stock_returns.rolling(window=period).corr(index_returns)
                stock_df[f'corr_{index.replace("^", "").replace("-", "_")}'] = correlation
        
        return stock_df
    
    def calculate_sector_correlations(self, stock_data: Dict[str, pd.DataFrame], 
                                    symbol: str) -> pd.DataFrame:
        """Calculate correlations with sector ETFs."""
        if symbol not in stock_data:
            return pd.DataFrame()
        
        stock_df = stock_data[symbol].copy()
        sector_etfs = self.config['data']['sector_etfs']
        
        for etf in sector_etfs:
            if etf in stock_data:
                etf_data = stock_data[etf]
                
                # Align dates
                common_dates = stock_df.index.intersection(etf_data.index)
                if len(common_dates) < self.correlation_config['sector_corr_period']:
                    continue
                
                # Calculate rolling correlation
                period = self.correlation_config['sector_corr_period']
                stock_returns = stock_df.loc[common_dates, 'Returns']
                etf_returns = etf_data.loc[common_dates, 'Returns']
                
                correlation = stock_returns.rolling(window=period).corr(etf_returns)
                stock_df[f'sector_corr_{etf}'] = correlation
        
        return stock_df
    
    def calculate_peer_correlations(self, stock_data: Dict[str, pd.DataFrame], 
                                  symbol: str) -> pd.DataFrame:
        """Calculate correlations with peer stocks."""
        if symbol not in stock_data:
            return pd.DataFrame()
        
        stock_df = stock_data[symbol].copy()
        symbols = self.config['data']['symbols']
        
        # Calculate average correlation with other stocks
        peer_correlations = []
        
        for peer_symbol in symbols:
            if peer_symbol == symbol or peer_symbol not in stock_data:
                continue
            
            peer_data = stock_data[peer_symbol]
            
            # Align dates
            common_dates = stock_df.index.intersection(peer_data.index)
            if len(common_dates) < self.correlation_config['peer_corr_period']:
                continue
            
            # Calculate rolling correlation
            period = self.correlation_config['peer_corr_period']
            stock_returns = stock_df.loc[common_dates, 'Returns']
            peer_returns = peer_data.loc[common_dates, 'Returns']
            
            correlation = stock_returns.rolling(window=period).corr(peer_returns)
            peer_correlations.append(correlation)
        
        if peer_correlations:
            # Average peer correlation
            peer_corr_df = pd.concat(peer_correlations, axis=1)
            stock_df['avg_peer_correlation'] = peer_corr_df.mean(axis=1)
            stock_df['peer_correlation_std'] = peer_corr_df.std(axis=1)
        
        return stock_df


class LearnableTemporalWeights(nn.Module):
    """Learnable temporal relevance parameters for news events."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.temporal_config = config['features']['temporal']
        
        self.max_decay_days = self.temporal_config['max_decay_days']
        self.num_event_categories = self.temporal_config['event_weight_categories']
        
        # Learnable parameters
        self.event_weights = nn.Parameter(torch.ones(self.num_event_categories))
        self.decay_rate = nn.Parameter(torch.tensor(self.temporal_config['initial_decay_rate']))
        
    def calculate_temporal_weights(self, days_ago: torch.Tensor, 
                                 event_categories: torch.Tensor) -> torch.Tensor:
        """
        Calculate temporal weights for news events.
        
        Args:
            days_ago: (batch_size, seq_len) - Days since each news event
            event_categories: (batch_size, seq_len) - Event category indices
            
        Returns:
            weights: (batch_size, seq_len) - Temporal weights for each event
        """
        # Exponential decay based on days
        time_decay = torch.exp(-self.decay_rate * days_ago)
        
        # Event importance weights
        event_importance = self.event_weights[event_categories]
        
        # Combined temporal weights
        temporal_weights = time_decay * event_importance
        
        return temporal_weights
    
    def forward(self, embeddings: torch.Tensor, days_ago: torch.Tensor, 
                event_categories: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal weights to embeddings.
        
        Args:
            embeddings: (batch_size, seq_len, embedding_dim) - News embeddings
            days_ago: (batch_size, seq_len) - Days since each news event
            event_categories: (batch_size, seq_len) - Event category indices
            
        Returns:
            weighted_embeddings: (batch_size, seq_len, embedding_dim) - Temporally weighted embeddings
        """
        temporal_weights = self.calculate_temporal_weights(days_ago, event_categories)
        
        # Apply weights to embeddings
        weighted_embeddings = embeddings * temporal_weights.unsqueeze(-1)
        
        return weighted_embeddings


class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.technical_indicators = TechnicalIndicators(config)
        self.market_context = MarketContextFeatures(config)
        self.temporal_weights = LearnableTemporalWeights(config)
        
    def create_target_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Create target labels for prediction."""
        # Calculate future returns
        df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Create directional labels
        threshold = 0.005  # 0.5% threshold for neutral class
        
        conditions = [
            df['future_return'] > threshold,      # Up
            df['future_return'] < -threshold,     # Down
        ]
        choices = [2, 0]  # Up=2, Down=0, Neutral=1 (default)
        
        df['target'] = np.select(conditions, choices, default=1)
        
        # Remove the last `horizon` rows (no target available)
        df = df.iloc[:-horizon]
        
        return df
    
    def align_news_with_prices(self, price_data: pd.DataFrame, 
                             news_data: List[Dict]) -> pd.DataFrame:
        """Align news data with price data by date."""
        if not news_data:
            # Return price data with empty news features
            price_data['daily_sentiment'] = 0.0
            price_data['news_count'] = 0
            price_data['sentiment_std'] = 0.0
            return price_data
        
        # Convert news to DataFrame
        news_df = pd.DataFrame(news_data)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Aggregate news by date
        daily_news = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_probs': lambda x: np.mean(np.stack(x), axis=0) if len(x) > 0 else np.array([0.33, 0.34, 0.33])
        }).reset_index()
        
        # Flatten column names
        daily_news.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_probs']
        daily_news['sentiment_std'] = daily_news['sentiment_std'].fillna(0)
        
        # Convert price data index to date
        price_data = price_data.copy()
        price_data['date'] = price_data.index.date
        
        # Merge with price data
        merged_data = price_data.merge(daily_news, on='date', how='left')
        
        # Fill missing values
        merged_data['sentiment_mean'] = merged_data['sentiment_mean'].fillna(0)
        merged_data['sentiment_std'] = merged_data['sentiment_std'].fillna(0)
        merged_data['news_count'] = merged_data['news_count'].fillna(0)
        
        # Set index back to datetime
        merged_data = merged_data.set_index(price_data.index)
        merged_data = merged_data.drop('date', axis=1)
        
        # Rename for consistency
        merged_data = merged_data.rename(columns={
            'sentiment_mean': 'daily_sentiment',
            'sentiment_std': 'sentiment_volatility'
        })
        
        return merged_data
    
    def engineer_features(self, stock_data: Dict[str, pd.DataFrame], 
                         news_data: Dict[str, List[Dict]], 
                         symbol: str) -> pd.DataFrame:
        """Engineer all features for a single symbol."""
        if symbol not in stock_data:
            logger.error(f"No stock data found for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Engineering features for {symbol}")
        
        # Start with price data
        df = stock_data[symbol].copy()
        
        # Calculate technical indicators
        df = self.technical_indicators.calculate_all_indicators(df)
        
        # Add market context features
        df = self.market_context.calculate_market_correlations(stock_data, symbol)
        df = self.market_context.calculate_sector_correlations(stock_data, symbol)
        df = self.market_context.calculate_peer_correlations(stock_data, symbol)
        
        # Align and add news features
        symbol_news = news_data.get(symbol, [])
        df = self.align_news_with_prices(df, symbol_news)
        
        # Create target labels
        df = self.create_target_labels(df)
        
        # Remove rows with NaN values (keep some for initial indicators)
        initial_rows = max(
            max(self.config['features']['technical']['sma_periods']),
            self.config['features']['correlation']['market_corr_period']
        )
        
        df = df.iloc[initial_rows:]
        
        logger.info(f"Generated {len(df.columns)} features for {symbol}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (excluding target and metadata)."""
        exclude_cols = ['target', 'Symbol', 'future_return']
        
        # This would normally be determined from actual data
        # For now, return a comprehensive list based on configuration
        feature_names = []
        
        # Technical indicators
        for period in self.config['features']['technical']['sma_periods']:
            feature_names.extend([f'sma_{period}', f'sma_{period}_ratio'])
        
        for period in self.config['features']['technical']['ema_periods']:
            feature_names.extend([f'ema_{period}', f'ema_{period}_ratio'])
        
        # Add other feature categories...
        # This is a simplified version - in practice, you'd get this from actual data
        
        return feature_names


def main():
    """Test the feature engineering functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 102 + np.random.randn(len(dates)).cumsum(),
        'Low': 98 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Test feature engineering
    engineer = FeatureEngineer(config)
    
    # Test technical indicators
    technical_data = engineer.technical_indicators.calculate_all_indicators(sample_data)
    print(f"Technical indicators: {technical_data.shape}")
    
    # Test temporal weights
    temporal_weights = engineer.temporal_weights
    print(f"Temporal weights initialized: {temporal_weights.event_weights.shape}")


if __name__ == "__main__":
    main()