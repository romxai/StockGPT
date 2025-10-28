"""
Data collection module for stock prices, news, and market context.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Optional, Tuple
import os
import pickle
import json # Added for saving raw news

logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects stock price data using yfinance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['data']['symbols']
        self.market_indices = config['data']['market_indices']
        self.sector_etfs = config['data']['sector_etfs']
        self.history_days = config['data']['history_days']
        
        # Define and create all data paths
        self.cache_dir = config['data']['cache_path']
        self.raw_data_path = config['data']['raw_data_path']
        self.raw_stock_dir = os.path.join(self.raw_data_path, 'stocks')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.raw_stock_dir, exist_ok=True)
        
    def collect_stock_data(self, symbol: str, period: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """Collect OHLCV data for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                data = ticker.history(period=period)
            elif start and end:
                data = ticker.history(start=start, end=end)
            else:
                # Default to history_days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.history_days)
                data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Clean and standardize data
            data = data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
            data.index.name = 'Date'
            data['Symbol'] = symbol
            
            # Calculate additional basic features
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Price_Change'] = data['Close'] - data['Open']
            data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Low']
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            
            logger.info(f"Collected {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_symbols(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Collect data for all configured symbols."""
        cache_file = os.path.join(self.cache_dir, 'stock_data.pkl')
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("Loaded stock data from cache")
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
        
        all_data = {}
        all_symbols = self.symbols + self.market_indices + self.sector_etfs
        
        for symbol in all_symbols:
            logger.info(f"Collecting data for {symbol}")
            data = self.collect_stock_data(symbol)
            if not data.empty:
                all_data[symbol] = data
                
                # --- Save raw stock data ---
                try:
                    raw_file_path = os.path.join(self.raw_stock_dir, f"{symbol}.csv")
                    data.to_csv(raw_file_path)
                    logger.debug(f"Saved raw stock data to {raw_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save raw stock data for {symbol}: {e}")
                # ---------------------------------
                
            time.sleep(0.1)  # Rate limiting
        
        # Cache the combined data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_data, f)
            logger.info(f"Cached combined stock data to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching combined stock data: {str(e)}")
        
        return all_data
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information and metadata."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            return {'symbol': symbol}


class NewsDataCollector:
    """Collects financial news from Yahoo Finance RSS feeds."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['data']['symbols']
        
        # Define and create all data paths
        self.cache_dir = config['data']['cache_path']
        self.raw_data_path = config['data']['raw_data_path']
        self.raw_news_dir = os.path.join(self.raw_data_path, 'news')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.raw_news_dir, exist_ok=True)
        
        # Yahoo Finance RSS feed URLs
        self.rss_urls = {
            'general': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'markets': 'https://feeds.finance.yahoo.com/rss/2.0/category-stocks',
            'tech': 'https://feeds.finance.yahoo.com/rss/2.0/category-tech',
        }
        
        # Added more keywords to search for
        self.rss_keywords = [
            "stocks", "market", "finance", "earnings", "technology", "economy"
        ]
        
        # Headers for web requests (kept for RSS)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_rss_news(self, symbol: str = None, keyword: str = None) -> List[Dict]:
        """Collect news from Yahoo Finance RSS feeds by symbol, category, or keyword."""
        all_news = []
        
        urls_to_fetch = {}

        if symbol:
            # Construct symbol-specific URL
            urls_to_fetch[f'symbol_{symbol}'] = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
        elif keyword:
            # Construct keyword-specific URL (searches news)
            urls_to_fetch[f'keyword_{keyword}'] = f"https://feeds.finance.yahoo.com/rss/v1/finance/News?query={keyword}"
        else:
            # Use predefined categories
            urls_to_fetch = self.rss_urls

        try:
            for category, url in urls_to_fetch.items():
                logger.debug(f"Fetching RSS feed from: {url}")
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    news_item = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'category': category,
                        'symbol': symbol if symbol else 'general',
                        'source': 'yahoo_rss'
                    }
                    
                    # Parse published date
                    try:
                        if 'published_parsed' in entry:
                            news_item['date'] = datetime(*entry.published_parsed[:6]).isoformat()
                        else:
                            news_item['date'] = datetime.now().isoformat()
                    except:
                        news_item['date'] = datetime.now().isoformat()
                    
                    all_news.append(news_item)
                
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting RSS news: {str(e)}")
        
        return all_news
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats from news sources."""
        if not date_str:
            return datetime.now()
        
        # Common date patterns
        patterns = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%B %d, %Y',
            '%b %d, %Y'
        ]
        
        for pattern in patterns:
            try:
                return datetime.strptime(date_str, pattern)
            except ValueError:
                continue
        
        # If parsing fails, return current time
        return datetime.now()
    
    def collect_all_news(self, use_cache: bool = True) -> Dict[str, List[Dict]]:
        """Collect news for all symbols."""
        cache_file = os.path.join(self.cache_dir, 'news_data.pkl')
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("Loaded news data from cache")
                # Convert date strings back to datetime objects
                for symbol, news_list in cached_data.items():
                    for item in news_list:
                        if 'date' in item and isinstance(item['date'], str):
                            item['date'] = datetime.fromisoformat(item['date'])
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading news cache: {str(e)}")
        
        all_news = {}
        
        # Expanded general news collection
        logger.info("Collecting general market news from categories...")
        general_news = self.collect_rss_news(symbol=None, keyword=None)
        
        logger.info("Collecting general market news from keywords...")
        for keyword in self.rss_keywords:
            logger.debug(f"Fetching keyword: {keyword}")
            general_news.extend(self.collect_rss_news(symbol=None, keyword=keyword))
        
        all_news['general'] = self._deduplicate_news(general_news)
        
        # --- Save raw general news ---
        try:
            raw_general_dir = os.path.join(self.raw_news_dir, 'general')
            os.makedirs(raw_general_dir, exist_ok=True)
            raw_file_path = os.path.join(raw_general_dir, 'rss_general_combined.json')
            with open(raw_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_news['general'], f, indent=2, default=str, ensure_ascii=False)
            logger.debug(f"Saved raw general news to {raw_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save raw general news: {e}")
        # ----------------------------------

        # Collect symbol-specific news
        for symbol in self.symbols:
            logger.info(f"Collecting news for {symbol}")
            
            # RSS news for symbol
            rss_news = self.collect_rss_news(symbol=symbol)
            
            # --- Save raw symbol news ---
            try:
                raw_symbol_dir = os.path.join(self.raw_news_dir, symbol)
                os.makedirs(raw_symbol_dir, exist_ok=True)
                
                rss_file_path = os.path.join(raw_symbol_dir, 'rss_news.json')
                with open(rss_file_path, 'w', encoding='utf-8') as f:
                    json.dump(rss_news, f, indent=2, default=str, ensure_ascii=False)
                
                logger.debug(f"Saved raw news for {symbol} to {raw_symbol_dir}")
            except Exception as e:
                logger.warning(f"Failed to save raw news for {symbol}: {e}")
            # -----------------------------------
            
            symbol_news = self._deduplicate_news(rss_news)
            all_news[symbol] = symbol_news
            time.sleep(1)  # Rate limiting between symbols
        
        # Cache the combined data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_news, f)
            logger.info(f"Cached combined news data to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching combined news data: {str(e)}")
        
        # Convert date strings to datetime objects for in-memory use
        for symbol, news_list in all_news.items():
            for item in news_list:
                if 'date' in item and isinstance(item['date'], str):
                    try:
                        item['date'] = datetime.fromisoformat(item['date'])
                    except ValueError:
                        logger.warning(f"Could not parse date {item['date']}, using now()")
                        item['date'] = datetime.now()


        return all_news
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on title similarity."""
        unique_news = []
        seen_titles = set()
        
        for news in news_list:
            title = news.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        return unique_news


class MarketContextCollector:
    """Collects additional market context data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = config['data']['cache_path']
        self.raw_data_path = config['data']['raw_data_path']
        self.raw_market_dir = os.path.join(self.raw_data_path, 'market_context')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.raw_market_dir, exist_ok=True)
    
    def collect_economic_indicators(self) -> pd.DataFrame:
        """Collect economic indicators (simplified version using yfinance)."""
        indicators = {
            '^TNX': '10_year_treasury',
            '^VIX': 'vix',
            'DX-Y.NYB': 'dollar_index'
        }
        
        all_indicators = pd.DataFrame()
        
        for symbol, name in indicators.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{self.config['data']['history_days']}d")
                
                if not data.empty:
                    # --- Save raw market data ---
                    raw_file_path = os.path.join(self.raw_market_dir, f"{name}.csv")
                    data.to_csv(raw_file_path)
                    # ---------------------------------

                    indicator_data = pd.DataFrame({
                        f'{name}_close': data['Close'],
                        f'{name}_volume': data['Volume']
                    })
                    
                    if all_indicators.empty:
                        all_indicators = indicator_data
                    else:
                        all_indicators = all_indicators.join(indicator_data, how='outer')
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error collecting {name}: {str(e)}")
        
        return all_indicators
    
    def collect_sector_performance(self) -> pd.DataFrame:
        """Collect sector ETF performance data."""
        sector_data = pd.DataFrame()
        
        for etf in self.config['data']['sector_etfs']:
            try:
                ticker = yf.Ticker(etf)
                data = ticker.history(period=f"{self.config['data']['history_days']}d")
                
                if not data.empty:
                    # --- Save raw market data ---
                    raw_file_path = os.path.join(self.raw_market_dir, f"{etf}_sector.csv")
                    data.to_csv(raw_file_path)
                    # ---------------------------------
                    
                    etf_data = pd.DataFrame({
                        f'{etf}_close': data['Close'],
                        f'{etf}_volume': data['Volume'],
                        f'{etf}_returns': data['Close'].pct_change()
                    })
                    
                    if sector_data.empty:
                        sector_data = etf_data
                    else:
                        sector_data = sector_data.join(etf_data, how='outer')
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error collecting sector data for {etf}: {str(e)}")
        
        return sector_data


def main():
    """Test the data collection functionality."""
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test stock data collection
    stock_collector = StockDataCollector(config)
    stock_data = stock_collector.collect_all_symbols(use_cache=False) # Disable cache for testing
    print(f"Collected stock data for {len(stock_data)} symbols")
    
    # Test news collection
    news_collector = NewsDataCollector(config)
    news_data = news_collector.collect_all_news(use_cache=False) # Disable cache for testing
    print(f"Collected news data for {len(news_data)} categories/symbols")
    
    # Test market context collection
    context_collector = MarketContextCollector(config)
    economic_data = context_collector.collect_economic_indicators()
    sector_data = context_collector.collect_sector_performance()
    print(f"Collected economic indicators: {economic_data.shape}")
    print(f"Collected sector data: {sector_data.shape}")


if __name__ == "__main__":
    main()