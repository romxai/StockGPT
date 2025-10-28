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

logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects stock price data using yfinance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['data']['symbols']
        self.market_indices = config['data']['market_indices']
        self.sector_etfs = config['data']['sector_etfs']
        self.history_days = config['data']['history_days']
        self.cache_dir = config['data']['cache_path']
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
            time.sleep(0.1)  # Rate limiting
        
        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_data, f)
            logger.info("Cached stock data")
        except Exception as e:
            logger.warning(f"Error caching data: {str(e)}")
        
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
    """Collects financial news from Yahoo Finance RSS feeds and web scraping."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['data']['symbols']
        self.cache_dir = config['data']['cache_path']
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Yahoo Finance RSS feed URLs
        self.rss_urls = {
            'general': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'markets': 'https://feeds.finance.yahoo.com/rss/2.0/category-stocks',
            'tech': 'https://feeds.finance.yahoo.com/rss/2.0/category-tech',
        }
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_rss_news(self, symbol: str = None) -> List[Dict]:
        """Collect news from Yahoo Finance RSS feeds."""
        all_news = []
        
        try:
            for category, url in self.rss_urls.items():
                if symbol:
                    # Construct symbol-specific URL
                    symbol_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
                    feed = feedparser.parse(symbol_url)
                else:
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
                            news_item['date'] = datetime(*entry.published_parsed[:6])
                        else:
                            news_item['date'] = datetime.now()
                    except:
                        news_item['date'] = datetime.now()
                    
                    all_news.append(news_item)
                
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting RSS news: {str(e)}")
        
        return all_news
    
    def collect_yahoo_finance_news(self, symbol: str) -> List[Dict]:
        """Scrape news from Yahoo Finance stock page."""
        news_items = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles (structure may change, so we use multiple selectors)
            news_containers = soup.find_all(['div', 'li'], {'class': lambda x: x and 'news' in x.lower()})
            
            for container in news_containers[:20]:  # Limit to 20 articles
                try:
                    title_elem = container.find(['h3', 'h4', 'a'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                    
                    # Try to find summary/description
                    summary_elem = container.find(['p', 'div'], {'class': lambda x: x and ('summary' in x.lower() or 'desc' in x.lower())})
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    
                    # Try to find date
                    date_elem = container.find(['time', 'span'], {'class': lambda x: x and 'date' in x.lower()})
                    date_str = date_elem.get_text(strip=True) if date_elem else ''
                    
                    if title:
                        news_item = {
                            'title': title,
                            'summary': summary,
                            'link': f"https://finance.yahoo.com{link}" if link.startswith('/') else link,
                            'date': self._parse_date(date_str),
                            'symbol': symbol,
                            'source': 'yahoo_finance'
                        }
                        news_items.append(news_item)
                        
                except Exception as e:
                    logger.debug(f"Error parsing news item: {str(e)}")
                    continue
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance news for {symbol}: {str(e)}")
        
        return news_items
    
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
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading news cache: {str(e)}")
        
        all_news = {}
        
        # Collect general market news
        logger.info("Collecting general market news")
        general_news = self.collect_rss_news()
        all_news['general'] = general_news
        
        # Collect symbol-specific news
        for symbol in self.symbols:
            logger.info(f"Collecting news for {symbol}")
            
            # RSS news for symbol
            rss_news = self.collect_rss_news(symbol)
            
            # Yahoo Finance news for symbol
            yahoo_news = self.collect_yahoo_finance_news(symbol)
            
            # Combine and deduplicate
            symbol_news = rss_news + yahoo_news
            symbol_news = self._deduplicate_news(symbol_news)
            
            all_news[symbol] = symbol_news
            time.sleep(1)  # Rate limiting between symbols
        
        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_news, f)
            logger.info("Cached news data")
        except Exception as e:
            logger.warning(f"Error caching news data: {str(e)}")
        
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
        os.makedirs(self.cache_dir, exist_ok=True)
    
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
    stock_data = stock_collector.collect_all_symbols()
    print(f"Collected stock data for {len(stock_data)} symbols")
    
    # Test news collection
    news_collector = NewsDataCollector(config)
    news_data = news_collector.collect_all_news()
    print(f"Collected news data for {len(news_data)} categories/symbols")
    
    # Test market context collection
    context_collector = MarketContextCollector(config)
    economic_data = context_collector.collect_economic_indicators()
    sector_data = context_collector.collect_sector_performance()
    print(f"Collected economic indicators: {economic_data.shape}")
    print(f"Collected sector data: {sector_data.shape}")


if __name__ == "__main__":
    main()