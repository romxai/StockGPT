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
import re
import collections
from tqdm import tqdm

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
                # Default to February 2023 till now for consistency with news data
                end_date = datetime.now()
                start_date = datetime(2023, 2, 1)  # Same as news data
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
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _generate_comprehensive_keywords(self, symbol: str) -> List[str]:
        """
        Generate focused keywords for news search based on the symbol.
        Optimized for relevance and speed - reduced keyword count for faster collection.
        """
        # Core keywords - most important and relevant first
        keywords = [
            symbol,  # Raw ticker symbol
            f"{symbol} stock",
            f"{symbol} earnings",
            f"{symbol} quarterly"
        ]
        
        # Focused symbol-specific keywords - only the most relevant ones
        symbol_specific = {
            'AAPL': [
                'Apple', 'iPhone', 'Tim Cook', 'Apple earnings',
                'Apple stock', 'iPad', 'Mac', 'Apple services',
                'Apple event', 'iOS'
            ],
            'AMZN': [
                'Amazon', 'AWS', 'Amazon Prime', 'Jeff Bezos',
                'Andy Jassy', 'Amazon earnings', 'Amazon stock',
                'cloud computing', 'e-commerce', 'Prime Day'
            ],
            'TSLA': [
                'Tesla', 'Elon Musk', 'Tesla earnings', 'Tesla stock',
                'Model 3', 'Model Y', 'Tesla deliveries', 'electric vehicle',
                'Cybertruck', 'Gigafactory'
            ],
            'NVDA': [
                'NVIDIA', 'Jensen Huang', 'NVIDIA earnings', 'NVIDIA stock',
                'GPU', 'AI chips', 'data center', 'GeForce',
                'artificial intelligence', 'semiconductor'
            ]
        }
        
        # Add focused symbol-specific keywords
        if symbol in symbol_specific:
            keywords.extend(symbol_specific[symbol])
        
        # Essential financial terms only
        essential_terms = [
            f"{symbol} earnings report",
            f"{symbol} revenue",
            f"{symbol} analyst",
            f"{symbol} price target"
        ]
        keywords.extend(essential_terms)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword.lower() not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword.lower())
        
        return unique_keywords
    
    def collect_historical_google_news(self,
                                      symbol: str,
                                      start_date: str,
                                      end_date: str,
                                      target_articles: Optional[int] = None,
                                      max_per_day: int = 3) -> List[Dict]:
        """
        Collect historical news from Google News archives using multiple search strategies.
        Uses both combined queries and individual keyword searches for comprehensive coverage.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            target_articles: Desired number of articles (None for no limit)
            max_per_day: Maximum articles to keep from a single day

        Returns:
            List of article dictionaries
        """
        articles = []
        day_counter = collections.defaultdict(int)

        # Get comprehensive keywords
        keywords = self._generate_comprehensive_keywords(symbol)
        
        # Convert dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Please use YYYY-MM-DD.")
            return []

        # Use monthly chunks - go forward from start_dt to end_dt (chronological order)
        total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
        current_start_date = start_dt

        # Progress bar for months
        month_progress = tqdm(total=total_months, desc=f"Collecting {symbol} news ({start_date} to {end_date})", unit="month", leave=False)
        
        # Search strategies to try in order
        search_strategies = []
        
        # Strategy 1: Top priority keywords (company name, ticker, earnings)
        priority_keywords = [k for k in keywords if any(term in k.lower() 
                           for term in [symbol.lower(), 'earnings', keywords[1].lower()])][:5]
        if priority_keywords:
            search_strategies.append(("Priority", " OR ".join([f'"{k}"' for k in priority_keywords])))
        
        # Strategy 2: Combined all keywords (fallback)
        search_strategies.append(("Combined", " OR ".join([f'"{k}"' for k in keywords[:10]])))
        
        # Strategy 3: Individual high-value keywords (if we still need more articles)
        top_individual = keywords[:3]  # Top 3 most relevant keywords
        for keyword in top_individual:
            search_strategies.append(("Individual", f'"{keyword}"'))

        while current_start_date < end_dt and (target_articles is None or len(articles) < target_articles):
            # Calculate month chunk - go forward chronologically
            # Get the last day of the current month
            if current_start_date.month == 12:
                next_month = current_start_date.replace(year=current_start_date.year + 1, month=1, day=1)
            else:
                next_month = current_start_date.replace(month=current_start_date.month + 1, day=1)
            
            chunk_end_dt = min(next_month - timedelta(days=1), end_dt)
            chunk_start_str = current_start_date.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end_dt.strftime('%Y-%m-%d')
            
            month_progress.set_postfix({"Articles": len(articles), "Month": chunk_start_str[:7]})

            # Try each search strategy for this month chunk
            for strategy_name, search_query in search_strategies:
                if target_articles is not None and len(articles) >= target_articles:
                    break
                    
                try:
                    # Google News RSS Search URL
                    search_url = "https://news.google.com/rss/search"
                    params = {
                        'q': f'{search_query} after:{chunk_start_str} before:{chunk_end_str}',
                        'hl': 'en-US',
                        'gl': 'US',
                        'ceid': 'US:en'
                    }

                    response = self.session.get(search_url, params=params, timeout=20)
                    response.raise_for_status()

                    if response.status_code == 200:
                        feed = feedparser.parse(response.content)
                        week_articles_added = 0

                        for entry in feed.entries:
                            if target_articles is not None and len(articles) >= target_articles:
                                break
                                
                            pub_date = None
                            try:
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                                elif 'published' in entry:
                                    try: 
                                        pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
                                    except ValueError: 
                                        try:
                                            pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S GMT")
                                        except ValueError:
                                            pass
                            except Exception:
                                continue

                            # Filter by date range and daily limit
                            if pub_date and start_dt <= pub_date <= end_dt:
                                date_key = pub_date.date()
                                if day_counter[date_key] < max_per_day:
                                    title = entry.get('title', '').strip()
                                    link = entry.get('link', '')
                                    
                                    # Skip if title is empty or too short
                                    if len(title) < 10:
                                        continue
                                        
                                    # Check for duplicate titles (case-insensitive)
                                    is_duplicate = any(
                                        title.lower() == existing['title'].lower() 
                                        for existing in articles
                                    )
                                    
                                    if not is_duplicate:
                                        news_item = {
                                            'title': title,
                                            'summary': "",
                                            'link': link,
                                            'published': pub_date.isoformat(),
                                            'date': pub_date.isoformat(),
                                            'category': 'google_news_historical',
                                            'symbol': symbol,
                                            'source': 'google_news_historical'
                                        }
                                        
                                        articles.append(news_item)
                                        day_counter[date_key] += 1
                                        week_articles_added += 1

                        # If this strategy found articles, add small delay
                        if week_articles_added > 0:
                            time.sleep(0.5)

                except requests.exceptions.RequestException:
                    time.sleep(2)
                except Exception:
                    continue
                
                # Small delay between strategies
                time.sleep(0.3)

            # Move to next month
            if current_start_date.month == 12:
                current_start_date = current_start_date.replace(year=current_start_date.year + 1, month=1, day=1)
            else:
                current_start_date = current_start_date.replace(month=current_start_date.month + 1, day=1)
            month_progress.update(1)
            
            # Rate limiting between months
            time.sleep(1.2)

        month_progress.close()

        # Final deduplication and sorting
        if articles:
            df = pd.DataFrame(articles)
            # More thorough deduplication
            df['title_clean'] = df['title'].str.lower().str.strip()
            df = df.drop_duplicates(subset=['title_clean'], keep='first')
            df = df.drop(columns=['title_clean'])
            
            df['published_dt'] = pd.to_datetime(df['published'])
            df = df.sort_values('published_dt', ascending=False).reset_index(drop=True)

            # Convert back to list of dicts
            return df.to_dict('records')
        else:
            return []
    

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
        """Collect historical news for all symbols using Google News scraping."""
        cache_file = os.path.join(self.cache_dir, 'news_data.pkl')

        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info("Loaded news data from cache")
                # Convert date strings back to datetime objects if needed (handle potential load errors)
                for symbol, news_list in cached_data.items():
                    for item in news_list:
                        if 'date' in item and isinstance(item['date'], str):
                            try:
                                item['date'] = datetime.fromisoformat(item['date'])
                            except ValueError:
                                logger.warning(f"Cache load: Could not parse date {item['date']}, using now()")
                                item['date'] = datetime.now() # Fallback
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading news cache: {str(e)}. Re-fetching data.")

        all_news = {}
        all_news['general'] = [] # We are focusing on symbol-specific historical news

        # Determine date range - from February 2023 till now
        end_date = datetime.now()
        start_date_dt = datetime(2023, 2, 1)  # Start from February 2023
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Collect symbol-specific historical news
        for symbol in tqdm(self.symbols, desc="Collecting news for symbols", unit="symbol"):
            # Use the new historical scraping method
            # No cap on total articles, but limit to max 3 per day from Feb 2023 till now
            historical_news = self.collect_historical_google_news(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date_str,
                target_articles=None,  # No total cap
                max_per_day=3
            )

            # --- Save raw symbol news ---
            try:
                raw_symbol_dir = os.path.join(self.raw_news_dir, symbol)
                os.makedirs(raw_symbol_dir, exist_ok=True)
                # Save the scraped news
                scrape_file_path = os.path.join(raw_symbol_dir, 'google_historical_news.json')
                with open(scrape_file_path, 'w', encoding='utf-8') as f:
                    # Use default=str to handle datetime objects during JSON dump
                    json.dump(historical_news, f, indent=2, default=str, ensure_ascii=False)
                logger.debug(f"Saved raw historical news for {symbol} to {raw_symbol_dir}")
            except Exception as e:
                logger.warning(f"Failed to save raw news for {symbol}: {e}")
            # -----------------------------------

            all_news[symbol] = historical_news
            # Removed the 1-second sleep, as sleeps are inside the scraping function now

        # Cache the combined data
        try:
            # Ensure dates are strings before pickling
            data_to_cache = {}
            for symbol, news_list in all_news.items():
                data_to_cache[symbol] = []
                for item in news_list:
                    item_copy = item.copy()
                    if isinstance(item_copy.get('date'), datetime):
                         item_copy['date'] = item_copy['date'].isoformat() # Convert to string
                    data_to_cache[symbol].append(item_copy)

            with open(cache_file, 'wb') as f:
                pickle.dump(data_to_cache, f) # Use data_to_cache
            logger.info(f"Cached combined news data to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching combined news data: {str(e)}")

        # Convert date strings back to datetime objects for in-memory use (after caching)
        for symbol, news_list in all_news.items():
            for item in news_list:
                if 'date' in item and isinstance(item['date'], str):
                    try:
                        item['date'] = datetime.fromisoformat(item['date'])
                    except ValueError:
                        logger.warning(f"Could not parse date {item['date']} after fetch, using now()")
                        item['date'] = datetime.now() # Fallback

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
                # Use same date range as stock and news data (Feb 2023 - now)
                start_date = datetime(2023, 2, 1)
                end_date = datetime.now()
                data = ticker.history(start=start_date, end=end_date)
                
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
                # Use same date range as stock and news data (Feb 2023 - now)
                start_date = datetime(2023, 2, 1)
                end_date = datetime.now()
                data = ticker.history(start=start_date, end=end_date)
                
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