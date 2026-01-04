# AI Personal Expense Assistant

An intelligent Streamlit application for analyzing, forecasting, and querying personal expense data using machine learning and natural language processing.

## Features

### 📊 Dashboard
- Visualize expense trends over time
- Category-wise expense distribution
- Key metrics and statistics
- Recent transactions table

### 💬 AI Chat Interface
- Natural language query processing (rule-based, no API calls)
- Ask questions like:
  - "restaurant expenses in january"
  - "total expenses"
  - "food above 500"
- Returns filtered data and natural language summaries

### 🔮 Forecasting
- Prophet-based time series forecasting
- Predict future expenses for next 7-90 days
- Confidence intervals and visualizations
- Category-specific forecasts

### ⚠️ Anomaly Detection
- Statistical anomaly detection (mean + 2*std)
- Identifies unusual transactions
- Visual anomaly highlighting
- Anomaly statistics and analysis

### 🤖 Auto-Categorization
- XGBoost + TF-IDF based categorization
- Trains on your data automatically
- Categorizes uncategorized expenses
- Model persistence for future use

## Architecture

### Folder Structure

```
expenses ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── core/                       # Core processing modules
│   ├── preprocessing.py        # Data loading and cleaning
│   ├── categorization.py       # Expense categorization (XGBoost + TF-IDF)
│   ├── insights.py            # Statistical insights (placeholder)
│   └── anomalies.py           # Anomaly detection
│
├── ml/                         # Machine learning modules
│   ├── train_model.py         # Model training and persistence
│   └── forecasting.py         # Prophet-based forecasting
│
├── ai/                         # AI/NLP modules
│   └── vibe_engine.py         # Natural language query processing
│
├── data/                       # Data directory
│   └── sample.csv             # Sample expense data
│
└── models/                     # Saved models (created at runtime)
    ├── vectorizer.pkl         # TF-IDF vectorizer
    └── classifier.pkl         # XGBoost classifier
```

## Models Used

### 1. Expense Categorization (XGBoost + TF-IDF)
- **Algorithm**: XGBoost Classifier
- **Text Processing**: TF-IDF Vectorization (1000 features)
- **Training**: Automatic training on user's categorized data
- **Persistence**: Models saved as `vectorizer.pkl` and `classifier.pkl`

### 2. Expense Forecasting (Prophet)
- **Algorithm**: Facebook Prophet time series model
- **Input**: Daily aggregated expenses
- **Output**: Future expense predictions with confidence intervals
- **Use Case**: Budget planning and expense prediction

### 3. Anomaly Detection (Statistical)
- **Method**: Mean + 2 Standard Deviations threshold
- **Purpose**: Identify unusually high transactions
- **Output**: Flagged transactions with visualization

### 4. Natural Language Query Engine (Rule-Based)
- **Approach**: Pattern matching and keyword extraction
- **Features**:
  - Month detection (january, feb, etc.)
  - Category keyword mapping
  - Amount filtering (above, below, over, under)
  - Description keyword matching
- **No API calls**: Fully local, rule-based processing

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

## Usage

### Data Format

Your expense file (CSV or Excel) should contain:
- **date**: Transaction date (required)
- **description**: Transaction description (required)
- **amount**: Transaction amount (required)
- **category**: Expense category (optional, will be auto-categorized if missing)

### Sample Query Examples

**Time-based queries:**
- "expenses in january"
- "march 2024 expenses"
- "expenses last month"

**Category queries:**
- "restaurant expenses"
- "food expenses in january"
- "coffee expenses"

**Amount-based queries:**
- "expenses above 500"
- "transactions over 100"
- "food above 50"

**Combined queries:**
- "restaurant expenses in january"
- "food above 500 in march"
- "total expenses"

## Vibe-Coding Workflow

The "Vibe Engine" (natural language query processor) uses a rule-based approach:

1. **Query Normalization**: Convert to lowercase, extract keywords
2. **Pattern Matching**: 
   - Month detection using predefined month map
   - Category keyword mapping (restaurant → food, coffee → beverages, etc.)
   - Amount threshold extraction using regex patterns
3. **Data Filtering**: Apply filters sequentially to DataFrame
4. **Summary Generation**: Create natural language summary from results

### Keyword Mapping

The system recognizes common expense keywords:
- **Restaurant**: restaurant, dining, dinner, lunch, cafe
- **Food**: food, grocery, supermarket, market, groceries
- **Coffee**: coffee, starbucks, cafe, espresso
- **Transport**: uber, taxi, transport, gas, fuel, car
- **Shopping**: shopping, amazon, target, walmart, purchase
- **Entertainment**: netflix, movie, cinema, entertainment
- **Bills**: bill, utility, electric, water, internet, phone

## Future Enhancements

### Short-term Improvements
1. **Multi-file support**: Upload and merge multiple expense files
2. **Export functionality**: Export filtered results and forecasts
3. **Budget setting**: Set budgets and track against them
4. **Recurring expense detection**: Identify subscriptions and recurring charges
5. **Category management**: Manual category editing and merging

### Medium-term Enhancements
1. **Advanced NLP**: Integration with transformer models for better query understanding
2. **Custom category training**: User-provided training examples
3. **Multi-currency support**: Handle expenses in different currencies
4. **Receipt OCR**: Extract expense data from receipt images
5. **Bank integration**: Direct integration with bank APIs

### Long-term Vision
1. **Mobile app**: Native mobile application
2. **Multi-user support**: Family/household expense tracking
3. **AI-powered insights**: Proactive spending recommendations
4. **Integration ecosystem**: Connect with accounting software
5. **Advanced forecasting**: Multi-variate forecasting with external factors

## Technical Details

### Data Preprocessing Pipeline

1. **Column Normalization**: Standardize column names (case-insensitive matching)
2. **Amount Cleaning**: Remove currency symbols, handle commas, convert to numeric
3. **Date Parsing**: Flexible date parsing with pandas
4. **Missing Value Handling**: 
   - Amount/date: Drop rows
   - Category: Set to "unknown"
   - Description: Use "uncategorized" or merge text columns
5. **Deduplication**: Remove duplicate transactions
6. **Feature Engineering**: Extract date features (year, month, day of week)

### Model Training

- **Trigger**: Automatic training when categorized data is available
- **Minimum Data**: Requires at least 10 categorized examples
- **Training Data**: Excludes "unknown" category entries
- **Persistence**: Models saved to `models/` directory
- **Inference**: Automatic categorization of "unknown" expenses

### Forecasting Pipeline

1. **Data Aggregation**: Group expenses by date and sum amounts
2. **Prophet Training**: Fit Prophet model on daily totals
3. **Future Prediction**: Generate forecast for specified period
4. **Visualization**: Plot with confidence intervals

## Troubleshooting

### Common Issues

1. **"Not enough data to generate forecast"**
   - Solution: Ensure you have at least 2 data points with different dates

2. **"Model must be trained before prediction"**
   - Solution: The system will auto-train. Ensure you have some categorized data.

3. **"Could not find or infer amount column"**
   - Solution: Ensure your file has a column with numeric expense amounts

4. **Date parsing errors**
   - Solution: Ensure dates are in a recognizable format (YYYY-MM-DD recommended)

## License

This project is provided as-is for educational and personal use.

## Contributing

Feel free to fork, modify, and enhance this project for your needs!


