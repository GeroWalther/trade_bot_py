class FREDService:
    def __init__(self):
        self.indicators = {
            # Core Economic Indicators
            'GDP': 'GDP',                    # Gross Domestic Product
            'UNRATE': 'UNRATE',             # Unemployment Rate
            'CPIAUCSL': 'CPIAUCSL',         # Consumer Price Index (Inflation)
            'FEDFUNDS': 'FEDFUNDS',         # Federal Funds Rate
            
            # Additional Key Indicators
            'M2': 'M2',                     # Money Supply
            'INDPRO': 'INDPRO',             # Industrial Production
            'PCE': 'PCE',                   # Personal Consumption Expenditures
            'HOUST': 'HOUST',               # Housing Starts
            
            # International Trade
            'BOPGSTB': 'BOPGSTB',          # Trade Balance
            'DTWEXB': 'DTWEXB',            # Dollar Index
            
            # Market Specific
            'BAA10Y': 'BAA10Y',            # Corporate Bond Spread
            'T10Y2Y': 'T10Y2Y',            # Yield Curve
            'VIXCLS': 'VIXCLS',            # VIX Volatility Index
            
            # Commodity Related
            'PPIACO': 'PPIACO',            # Producer Price Index
            'DCOILWTICO': 'DCOILWTICO',    # Crude Oil Prices
            'GOLDAMGBD228NLBM': 'GOLDAMGBD228NLBM',  # Gold Prices
            
            # Regional Indicators
            'RGDPUS': 'RGDPUS',            # Real GDP by State
            'CPILFESL': 'CPILFESL',        # Core Inflation
            'RSAFS': 'RSAFS',              # Retail Sales
        } 