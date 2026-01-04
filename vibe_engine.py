"""
Natural Language Query Engine (Vibe Engine)
Processes natural language queries about expenses and returns results.
Rule-based implementation (no API calls).
"""

import pandas as pd
import re
from datetime import datetime


class VibeEngine:
    """Processes natural language queries about expenses."""
    
    def __init__(self, df):
        """
        Initialize the vibe engine with expense data.
        
        Args:
            df: Expense DataFrame
        """
        self.df = df.copy()
        
        # Month mapping
        self.month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # Keyword to category mapping
        self.keyword_map = {
            'restaurant': ['restaurant', 'dining', 'dinner', 'lunch', 'eat', 'food', 'cafe', 'café'],
            'food': ['food', 'grocery', 'supermarket', 'market', 'groceries', 'store', 'grocery store'],
            'coffee': ['coffee', 'starbucks', 'cafe', 'café', 'espresso'],
            'transport': ['uber', 'taxi', 'transport', 'gas', 'fuel', 'petrol', 'car', 'ride'],
            'shopping': ['shopping', 'amazon', 'target', 'walmart', 'store', 'purchase', 'buy'],
            'entertainment': ['netflix', 'movie', 'cinema', 'entertainment', 'streaming'],
            'bills': ['bill', 'utility', 'electric', 'water', 'internet', 'phone', 'subscription']
        }
    
    def generate_report_from_text(self, query, df):
        """
        Generate report from text query.
        
        Args:
            query: Natural language query string
            df: Expense DataFrame to query
            
        Returns:
            tuple: (filtered_dataframe, summary_string)
        """
        self.df = df.copy()
        
        # Parse query
        filters = self._parse_query(query.lower())
        
        # Apply filters
        result_df = self._execute_query(filters)
        
        # Generate summary
        summary = self._generate_summary(result_df, filters, query)
        
        return result_df, summary
    
    def _parse_query(self, query):
        """
        Parse query to extract filters and intent.
        
        Args:
            query: Natural language query (lowercase)
            
        Returns:
            dict: Parsed query components
        """
        filters = {
            'month': None,
            'year': None,
            'category_keywords': [],
            'min_amount': None,
            'max_amount': None,
            'description_keywords': []
        }
        
        # Detect month
        for month_name, month_num in self.month_map.items():
            if month_name in query:
                filters['month'] = month_num
                # Also try to extract year if mentioned
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    filters['year'] = int(year_match.group(1))
                break
        
        # Detect category keywords
        for category, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in query:
                    filters['category_keywords'].append(category)
                    filters['description_keywords'].extend(keywords)
                    break
        
        # Detect amount filters (above, below, over, under, more than, less than)
        amount_patterns = [
            (r'(?:above|over|more than|greater than)\s+(\d+)', 'min'),
            (r'(?:below|under|less than|lower than)\s+(\d+)', 'max'),
            (r'(\d+)\s*(?:and|to|\-)\s*above', 'min'),
        ]
        
        for pattern, filter_type in amount_patterns:
            match = re.search(pattern, query)
            if match:
                amount = float(match.group(1))
                if filter_type == 'min':
                    filters['min_amount'] = amount
                elif filter_type == 'max':
                    filters['max_amount'] = amount
        
        # Detect exact amount mentions
        amount_match = re.search(r'\$?(\d+(?:\.\d+)?)', query)
        if amount_match and not filters['min_amount'] and not filters['max_amount']:
            # If "above X" or "over X" wasn't found, look for standalone numbers
            if 'above' in query or 'over' in query or 'more' in query:
                filters['min_amount'] = float(amount_match.group(1))
            elif 'below' in query or 'under' in query or 'less' in query:
                filters['max_amount'] = float(amount_match.group(1))
        
        # Extract description keywords (words that might be in descriptions)
        words = query.split()
        common_words = {'in', 'on', 'the', 'and', 'or', 'of', 'for', 'with', 'from', 'to', 'a', 'an', 'expenses', 'expense', 'total', 'spent', 'spending'}
        description_words = [w for w in words if w not in common_words and len(w) > 2]
        filters['description_keywords'].extend(description_words)
        
        return filters
    
    def _execute_query(self, filters):
        """
        Execute the parsed query on the data.
        
        Args:
            filters: Parsed query dictionary
            
        Returns:
            pd.DataFrame: Query results
        """
        result_df = self.df.copy()
        
        # Filter by month
        if filters['month']:
            if 'date' in result_df.columns:
                if filters['year']:
                    result_df = result_df[
                        (result_df['date'].dt.month == filters['month']) &
                        (result_df['date'].dt.year == filters['year'])
                    ]
                else:
                    result_df = result_df[result_df['date'].dt.month == filters['month']]
        
        # Filter by category keywords
        if filters['category_keywords']:
            category_mask = result_df['category'].str.lower().isin([cat.lower() for cat in filters['category_keywords']])
            # Also check descriptions
            desc_mask = result_df['description'].str.lower().str.contains('|'.join(filters['category_keywords']), case=False, na=False)
            result_df = result_df[category_mask | desc_mask]
        
        # Filter by description keywords
        if filters['description_keywords']:
            keyword_pattern = '|'.join(filters['description_keywords'])
            desc_mask = result_df['description'].str.lower().str.contains(keyword_pattern, case=False, na=False)
            result_df = result_df[desc_mask]
        
        # Filter by amount
        if filters['min_amount']:
            result_df = result_df[result_df['amount'] >= filters['min_amount']]
        
        if filters['max_amount']:
            result_df = result_df[result_df['amount'] <= filters['max_amount']]
        
        return result_df.reset_index(drop=True)
    
    def _generate_summary(self, result_df, filters, original_query):
        """
        Generate a summary sentence from query results.
        
        Args:
            result_df: Filtered DataFrame
            filters: Applied filters
            original_query: Original user query
            
        Returns:
            str: Summary sentence
        """
        if len(result_df) == 0:
            return "No expenses found matching your query."
        
        total_amount = result_df['amount'].sum()
        transaction_count = len(result_df)
        
        # Build summary parts
        parts = []
        
        # Amount
        parts.append(f"You spent ${total_amount:,.2f}")
        
        # Category/description context
        if filters['category_keywords']:
            category_str = ', '.join(filters['category_keywords'][:2])
            parts.append(f"on {category_str}")
        elif 'total' in original_query.lower():
            parts.append("in total")
        else:
            parts.append("on expenses")
        
        # Time context
        if filters['month']:
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                          5: 'May', 6: 'June', 7: 'July', 8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            month_str = month_names[filters['month']]
            if filters['year']:
                parts.append(f"in {month_str} {filters['year']}")
            else:
                parts.append(f"in {month_str}")
        
        # Transaction count
        parts.append(f"across {transaction_count} transaction{'s' if transaction_count != 1 else ''}")
        
        summary = ' '.join(parts) + "."
        
        return summary

