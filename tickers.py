"""
Curated ticker universes for the Global Live Stock Tracker.

~100 large-cap names per market. Yahoo Finance symbol format.
For exhaustive index coverage (S&P 500, NIFTY 500, Hang Seng Composite,
STOXX 600, Nikkei 225, KOSPI 200), see `fetch_index_constituents` in app.py.
"""

USA: dict[str, str] = {
    # Mega-cap tech
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet A",
    "GOOG": "Alphabet C", "AMZN": "Amazon", "META": "Meta Platforms", "TSLA": "Tesla",
    "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe",
    "NFLX": "Netflix", "AMD": "AMD", "INTC": "Intel", "CSCO": "Cisco",
    "QCOM": "Qualcomm", "TXN": "Texas Instruments", "IBM": "IBM", "NOW": "ServiceNow",
    "INTU": "Intuit", "PYPL": "PayPal", "UBER": "Uber", "ABNB": "Airbnb",
    "SHOP": "Shopify", "SNOW": "Snowflake", "PLTR": "Palantir", "CRWD": "CrowdStrike",
    "MU": "Micron", "KLAC": "KLA", "LRCX": "Lam Research", "AMAT": "Applied Materials",
    "ASML": "ASML ADR", "TSM": "TSMC ADR", "SAP": "SAP ADR",
    # Financials
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "C": "Citigroup", "SCHW": "Charles Schwab", "AXP": "American Express",
    "BLK": "BlackRock", "BX": "Blackstone", "KKR": "KKR", "V": "Visa",
    "MA": "Mastercard", "FI": "Fiserv", "BRK-B": "Berkshire B", "SPGI": "S&P Global",
    "CME": "CME Group", "ICE": "Intercontinental Exchange", "USB": "US Bancorp", "PNC": "PNC Financial",
    # Healthcare
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "ABBV": "AbbVie",
    "MRK": "Merck", "PFE": "Pfizer", "TMO": "Thermo Fisher", "ABT": "Abbott",
    "DHR": "Danaher", "BMY": "Bristol-Myers", "AMGN": "Amgen", "GILD": "Gilead",
    "ISRG": "Intuitive Surgical", "VRTX": "Vertex", "REGN": "Regeneron", "CVS": "CVS Health",
    "ELV": "Elevance Health", "ZTS": "Zoetis", "MDT": "Medtronic", "BSX": "Boston Scientific",
    # Consumer
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "TJX": "TJX Companies", "LOW": "Lowe's",
    "DIS": "Disney", "BKNG": "Booking Holdings", "CMCSA": "Comcast", "T": "AT&T",
    "VZ": "Verizon", "TMUS": "T-Mobile US", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "MDLZ": "Mondelez", "PM": "Philip Morris", "MO": "Altria",
    # Industrials & Energy
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "EOG": "EOG Resources",
    "SLB": "SLB", "PSX": "Phillips 66", "MPC": "Marathon Petroleum", "OXY": "Occidental",
    "CAT": "Caterpillar", "DE": "Deere", "BA": "Boeing", "GE": "GE Aerospace",
    "RTX": "RTX Corp", "LMT": "Lockheed Martin", "HON": "Honeywell", "UPS": "UPS",
    "FDX": "FedEx", "UNP": "Union Pacific", "CSX": "CSX", "NSC": "Norfolk Southern",
    "ETN": "Eaton", "PH": "Parker Hannifin", "ITW": "Illinois Tool Works",
    # Utilities / Real Estate / Materials
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Company", "AEP": "American Electric",
    "AMT": "American Tower", "PLD": "Prologis", "EQIX": "Equinix", "SPG": "Simon Property",
    "LIN": "Linde", "APD": "Air Products", "FCX": "Freeport-McMoRan", "NEM": "Newmont",
}

HONG_KONG: dict[str, str] = {
    # Tech / Internet
    "0700.HK": "Tencent", "9988.HK": "Alibaba", "3690.HK": "Meituan", "1810.HK": "Xiaomi",
    "9618.HK": "JD.com", "9999.HK": "NetEase", "9888.HK": "Baidu", "1024.HK": "Kuaishou",
    "9961.HK": "Trip.com", "6618.HK": "JD Health", "6098.HK": "Country Garden Services",
    "2382.HK": "Sunny Optical", "2018.HK": "AAC Technologies", "0992.HK": "Lenovo",
    "0763.HK": "ZTE",
    # Financials
    "0005.HK": "HSBC", "1299.HK": "AIA Group", "0388.HK": "HKEX", "2318.HK": "Ping An",
    "0939.HK": "China Construction Bank", "1398.HK": "ICBC", "3988.HK": "Bank of China",
    "2388.HK": "BOC Hong Kong", "2628.HK": "China Life", "3328.HK": "Bank of Communications",
    "3968.HK": "China Merchants Bank", "0011.HK": "Hang Seng Bank", "2601.HK": "China Pacific Insurance",
    "1288.HK": "Agricultural Bank of China",
    # Telecom & Tech infra
    "0941.HK": "China Mobile", "0762.HK": "China Unicom", "0728.HK": "China Telecom",
    # Energy
    "0883.HK": "CNOOC", "0386.HK": "Sinopec", "0857.HK": "PetroChina", "1088.HK": "China Shenhua",
    # Consumer
    "1929.HK": "Chow Tai Fook", "0288.HK": "WH Group", "0291.HK": "China Resources Beer",
    "2020.HK": "ANTA Sports", "2331.HK": "Li Ning", "2313.HK": "Shenzhou International",
    "1044.HK": "Hengan", "2319.HK": "Mengniu Dairy", "0322.HK": "Tingyi",
    "9633.HK": "Nongfu Spring", "6862.HK": "Haidilao",
    # Property / REITs
    "0001.HK": "CK Hutchison", "0016.HK": "Sun Hung Kai Properties", "0017.HK": "New World Development",
    "0101.HK": "Hang Lung Properties", "0688.HK": "China Overseas Land", "0823.HK": "Link REIT",
    "1109.HK": "China Resources Land", "1113.HK": "CK Asset", "1997.HK": "Wharf REIC",
    "0066.HK": "MTR", "0003.HK": "Hong Kong & China Gas",
    # Industrials / Materials / Pharma
    "1211.HK": "BYD", "0175.HK": "Geely Auto", "1177.HK": "Sino Biopharm",
    "1093.HK": "CSPC Pharma", "2269.HK": "WuXi Biologics", "6160.HK": "BeiGene",
    "2899.HK": "Zijin Mining", "0267.HK": "CITIC", "0868.HK": "Xinyi Glass",
    "2688.HK": "ENN Energy", "1038.HK": "CK Infrastructure", "0027.HK": "Galaxy Entertainment",
    "1928.HK": "Sands China", "0151.HK": "Want Want China", "0656.HK": "Fosun International",
    "0019.HK": "Swire Pacific", "0066.HK": "MTR Corp", "1099.HK": "Sinopharm",
    "2007.HK": "Country Garden", "0688.HK": "China Overseas Land",
}

INDIA: dict[str, str] = {
    # IT
    "TCS.NS": "TCS", "INFY.NS": "Infosys", "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro", "TECHM.NS": "Tech Mahindra", "LTIM.NS": "LTIMindtree",
    "MPHASIS.NS": "Mphasis", "PERSISTENT.NS": "Persistent Systems", "COFORGE.NS": "Coforge",
    # Banks & Financials
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "AXISBANK.NS": "Axis Bank", "INDUSINDBK.NS": "IndusInd Bank",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv", "HDFCLIFE.NS": "HDFC Life",
    "SBILIFE.NS": "SBI Life", "ICICIGI.NS": "ICICI Lombard", "ICICIPRULI.NS": "ICICI Prudential",
    "SHRIRAMFIN.NS": "Shriram Finance", "CHOLAFIN.NS": "Cholamandalam", "MUTHOOTFIN.NS": "Muthoot Finance",
    "BANKBARODA.NS": "Bank of Baroda", "PNB.NS": "Punjab National Bank", "SBICARD.NS": "SBI Cards",
    # Energy / Materials
    "RELIANCE.NS": "Reliance Industries", "ONGC.NS": "ONGC", "BPCL.NS": "BPCL",
    "IOC.NS": "IOC", "COALINDIA.NS": "Coal India", "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel", "JINDALSTEL.NS": "Jindal Steel", "HINDALCO.NS": "Hindalco",
    "VEDL.NS": "Vedanta", "NMDC.NS": "NMDC", "SAIL.NS": "SAIL",
    # Auto
    "MARUTI.NS": "Maruti Suzuki", "M&M.NS": "Mahindra & Mahindra", "TATAMOTORS.NS": "Tata Motors",
    "EICHERMOT.NS": "Eicher Motors", "BAJAJ-AUTO.NS": "Bajaj Auto", "HEROMOTOCO.NS": "Hero MotoCorp",
    "TVSMOTOR.NS": "TVS Motor", "MOTHERSON.NS": "Motherson Sumi", "BOSCHLTD.NS": "Bosch",
    # Pharma & Healthcare
    "SUNPHARMA.NS": "Sun Pharma", "CIPLA.NS": "Cipla", "DRREDDY.NS": "Dr Reddy's",
    "DIVISLAB.NS": "Divi's Labs", "APOLLOHOSP.NS": "Apollo Hospitals", "BIOCON.NS": "Biocon",
    "LUPIN.NS": "Lupin", "AUROPHARMA.NS": "Aurobindo Pharma", "ZYDUSLIFE.NS": "Zydus Lifesciences",
    # FMCG / Consumer
    "HINDUNILVR.NS": "Hindustan Unilever", "ITC.NS": "ITC", "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia", "DABUR.NS": "Dabur", "MARICO.NS": "Marico",
    "GODREJCP.NS": "Godrej Consumer", "COLPAL.NS": "Colgate-Palmolive", "TATACONSUM.NS": "Tata Consumer",
    "VBL.NS": "Varun Beverages", "PAGEIND.NS": "Page Industries", "TITAN.NS": "Titan",
    # Capital goods / Cement / Infra
    "LT.NS": "Larsen & Toubro", "ULTRACEMCO.NS": "UltraTech Cement", "GRASIM.NS": "Grasim",
    "AMBUJACEM.NS": "Ambuja Cements", "SHREECEM.NS": "Shree Cement", "ACC.NS": "ACC",
    "SIEMENS.NS": "Siemens India", "ABB.NS": "ABB India", "HAVELLS.NS": "Havells",
    "HAL.NS": "Hindustan Aeronautics", "BEL.NS": "Bharat Electronics",
    # Telecom / Media
    "BHARTIARTL.NS": "Bharti Airtel", "IDEA.NS": "Vodafone Idea", "INDUSTOWER.NS": "Indus Towers",
    "ZEEL.NS": "Zee Entertainment", "NAUKRI.NS": "Info Edge",
    # Power / Utilities
    "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid", "TATAPOWER.NS": "Tata Power",
    "ADANIPOWER.NS": "Adani Power", "ADANIGREEN.NS": "Adani Green", "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports", "PFC.NS": "Power Finance Corp", "RECLTD.NS": "REC Limited",
    # Retail / Real Estate / Other
    "DMART.NS": "Avenue Supermarts", "DLF.NS": "DLF", "TRENT.NS": "Trent",
    "ZOMATO.NS": "Zomato", "ASIANPAINT.NS": "Asian Paints", "BERGEPAINT.NS": "Berger Paints",
    "PIDILITIND.NS": "Pidilite", "SRF.NS": "SRF", "UPL.NS": "UPL", "GAIL.NS": "GAIL",
    "IRCTC.NS": "IRCTC", "BAJAJHLDNG.NS": "Bajaj Holdings",
}

EUROPE: dict[str, str] = {
    # Germany (DAX)
    "SAP.DE": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz", "DTE.DE": "Deutsche Telekom",
    "MUV2.DE": "Munich Re", "BAS.DE": "BASF", "BMW.DE": "BMW", "MBG.DE": "Mercedes-Benz",
    "VOW3.DE": "Volkswagen Pref", "DBK.DE": "Deutsche Bank", "ADS.DE": "Adidas",
    "IFX.DE": "Infineon", "BAYN.DE": "Bayer", "RHM.DE": "Rheinmetall", "CON.DE": "Continental",
    "DPW.DE": "Deutsche Post", "EOAN.DE": "E.ON", "FRE.DE": "Fresenius", "HEN3.DE": "Henkel Pref",
    "HEI.DE": "Heidelberg Materials", "LIN.DE": "Linde", "MRK.DE": "Merck KGaA",
    "PUM.DE": "Puma", "RWE.DE": "RWE", "SHL.DE": "Siemens Healthineers", "SY1.DE": "Symrise",
    "VNA.DE": "Vonovia", "ZAL.DE": "Zalando", "DHL.DE": "DHL Group",
    # France (CAC 40)
    "MC.PA": "LVMH", "OR.PA": "L'Oreal", "RMS.PA": "Hermes", "SAN.PA": "Sanofi",
    "AIR.PA": "Airbus", "BNP.PA": "BNP Paribas", "ACA.PA": "Credit Agricole", "GLE.PA": "Societe Generale",
    "AI.PA": "Air Liquide", "BN.PA": "Danone", "CAP.PA": "Capgemini", "DG.PA": "Vinci",
    "EL.PA": "EssilorLuxottica", "KER.PA": "Kering", "LR.PA": "Legrand", "ML.PA": "Michelin",
    "ORA.PA": "Orange", "RI.PA": "Pernod Ricard", "SU.PA": "Schneider Electric", "TTE.PA": "TotalEnergies",
    "STLA.PA": "Stellantis", "PUB.PA": "Publicis", "VIV.PA": "Vivendi", "CS.PA": "AXA",
    # Switzerland
    "NESN.SW": "Nestle", "ROG.SW": "Roche", "NOVN.SW": "Novartis", "UBSG.SW": "UBS",
    "CFR.SW": "Richemont", "ABBN.SW": "ABB", "ZURN.SW": "Zurich Insurance",
    "GIVN.SW": "Givaudan", "SREN.SW": "Swiss Re", "LONN.SW": "Lonza",
    # UK (FTSE)
    "AZN.L": "AstraZeneca", "SHEL.L": "Shell", "BP.L": "BP", "GSK.L": "GSK",
    "ULVR.L": "Unilever", "HSBA.L": "HSBC UK", "BARC.L": "Barclays", "LLOY.L": "Lloyds",
    "NWG.L": "NatWest", "PRU.L": "Prudential", "DGE.L": "Diageo", "TSCO.L": "Tesco",
    "VOD.L": "Vodafone UK", "BATS.L": "British American Tobacco", "GLEN.L": "Glencore",
    "RIO.L": "Rio Tinto", "REL.L": "RELX", "NG.L": "National Grid", "LSEG.L": "LSE Group",
    "BHP.L": "BHP London", "STAN.L": "Standard Chartered", "AHT.L": "Ashtead",
    # Netherlands (AEX)
    "ASML.AS": "ASML", "ADYEN.AS": "Adyen", "PRX.AS": "Prosus", "INGA.AS": "ING",
    "AD.AS": "Ahold Delhaize", "PHIA.AS": "Philips", "WKL.AS": "Wolters Kluwer",
    "AGN.AS": "Aegon", "MT.AS": "ArcelorMittal", "RAND.AS": "Randstad", "DSM.AS": "DSM-Firmenich",
    "UNA.AS": "Unilever NL", "AKZA.AS": "AkzoNobel",
    # Spain
    "ITX.MC": "Inditex", "IBE.MC": "Iberdrola", "BBVA.MC": "BBVA", "SAN.MC": "Santander",
    "TEF.MC": "Telefonica", "REP.MC": "Repsol", "AENA.MC": "AENA",
    # Italy
    "ENEL.MI": "Enel", "ENI.MI": "Eni", "ISP.MI": "Intesa Sanpaolo", "UCG.MI": "UniCredit",
    "RACE.MI": "Ferrari", "STM.MI": "STMicro IT", "STLAM.MI": "Stellantis IT",
    # Denmark / Sweden / Belgium
    "NOVO-B.CO": "Novo Nordisk", "DSV.CO": "DSV", "MAERSK-B.CO": "Maersk",
    "ATCO-A.ST": "Atlas Copco", "VOLV-B.ST": "Volvo", "ASSA-B.ST": "ASSA ABLOY",
    "HM-B.ST": "H&M", "ERIC-B.ST": "Ericsson", "SAND.ST": "Sandvik",
    "ABI.BR": "AB InBev", "KBC.BR": "KBC", "SOLB.BR": "Solvay", "UCB.BR": "UCB",
}

ASIA_OTHER: dict[str, str] = {
    # Japan (Nikkei top)
    "7203.T": "Toyota", "6758.T": "Sony", "9984.T": "SoftBank Group", "6861.T": "Keyence",
    "8035.T": "Tokyo Electron", "8306.T": "Mitsubishi UFJ", "9432.T": "NTT", "9433.T": "KDDI",
    "6098.T": "Recruit Holdings", "4063.T": "Shin-Etsu Chemical", "6273.T": "SMC",
    "6367.T": "Daikin", "6501.T": "Hitachi", "6594.T": "Nidec", "6857.T": "Advantest",
    "6902.T": "Denso", "6981.T": "Murata", "7011.T": "Mitsubishi Heavy", "7267.T": "Honda",
    "7269.T": "Suzuki", "7270.T": "Subaru", "7741.T": "HOYA", "7751.T": "Canon",
    "8001.T": "Itochu", "8031.T": "Mitsui & Co", "8053.T": "Sumitomo Corp", "8058.T": "Mitsubishi Corp",
    "8316.T": "Sumitomo Mitsui Financial", "8411.T": "Mizuho Financial",
    "8801.T": "Mitsui Fudosan", "8802.T": "Mitsubishi Estate", "9020.T": "East Japan Railway",
    "9022.T": "Central Japan Railway", "9101.T": "Nippon Yusen", "9501.T": "TEPCO",
    "9503.T": "Kansai Electric", "9531.T": "Tokyo Gas", "9613.T": "NTT Data",
    "4502.T": "Takeda Pharma", "4503.T": "Astellas", "4519.T": "Chugai Pharma",
    "4543.T": "Terumo", "4568.T": "Daiichi Sankyo", "4661.T": "Oriental Land",
    "4901.T": "Fujifilm", "6178.T": "Japan Post", "6301.T": "Komatsu", "6326.T": "Kubota",
    "6503.T": "Mitsubishi Electric", "6752.T": "Panasonic", "6954.T": "Fanuc",
    "7733.T": "Olympus", "7974.T": "Nintendo", "8002.T": "Marubeni", "9434.T": "SoftBank Corp",
    # Korea (KOSPI)
    "005930.KS": "Samsung Electronics", "000660.KS": "SK Hynix", "005380.KS": "Hyundai Motor",
    "005490.KS": "POSCO Holdings", "006400.KS": "Samsung SDI", "012330.KS": "Hyundai Mobis",
    "028260.KS": "Samsung C&T", "035420.KS": "Naver", "035720.KS": "Kakao", "051910.KS": "LG Chem",
    "055550.KS": "Shinhan Financial", "105560.KS": "KB Financial", "207940.KS": "Samsung Biologics",
    "373220.KS": "LG Energy Solution", "000270.KS": "Kia", "086790.KS": "Hana Financial",
    "015760.KS": "KEPCO", "033780.KS": "KT&G", "066570.KS": "LG Electronics",
    "017670.KS": "SK Telecom", "030200.KS": "KT Corp", "032830.KS": "Samsung Life",
    "090430.KS": "Amorepacific", "096770.KS": "SK Innovation",
    # Taiwan
    "2330.TW": "TSMC", "2317.TW": "Hon Hai", "2454.TW": "MediaTek", "2308.TW": "Delta Electronics",
    "2382.TW": "Quanta", "2412.TW": "Chunghwa Telecom", "2881.TW": "Fubon Financial",
    "2882.TW": "Cathay Financial", "2884.TW": "E.SUN Financial", "2891.TW": "CTBC Financial",
    "2912.TW": "President Chain", "3008.TW": "Largan Precision", "3045.TW": "Taiwan Mobile",
    "3711.TW": "ASE Tech", "1303.TW": "Nan Ya Plastics", "1301.TW": "Formosa Plastics",
    "2002.TW": "China Steel", "2207.TW": "Hotai Motor", "2885.TW": "Yuanta Financial",
    # Singapore
    "D05.SI": "DBS", "U11.SI": "UOB", "O39.SI": "OCBC", "C09.SI": "City Developments",
    "C31.SI": "CapitaLand Invest", "C38U.SI": "CapitaLand Integrated", "J36.SI": "Jardine Matheson",
    "S58.SI": "SATS", "S68.SI": "SGX", "Z74.SI": "Singtel", "V03.SI": "Venture Corp",
    "M44U.SI": "Mapletree Logistics", "Y92.SI": "Thai Beverage", "G13.SI": "Genting Singapore",
    # Australia (ASX)
    "BHP.AX": "BHP", "CBA.AX": "Commonwealth Bank", "CSL.AX": "CSL", "NAB.AX": "NAB",
    "WBC.AX": "Westpac", "ANZ.AX": "ANZ", "FMG.AX": "Fortescue", "MQG.AX": "Macquarie Group",
    "WES.AX": "Wesfarmers", "WOW.AX": "Woolworths", "RIO.AX": "Rio Tinto AU",
    "GMG.AX": "Goodman Group", "TLS.AX": "Telstra", "WDS.AX": "Woodside Energy",
    "TCL.AX": "Transurban", "QBE.AX": "QBE Insurance", "ALL.AX": "Aristocrat Leisure",
    "STO.AX": "Santos", "NCM.AX": "Newcrest Mining", "ASX.AX": "ASX Limited",
}


MARKET_PRESETS: dict[str, dict[str, str]] = {
    "USA": USA,
    "Hong Kong": HONG_KONG,
    "India": INDIA,
    "Europe": EUROPE,
    "Asia (ex-HK/India)": ASIA_OTHER,
}


# ---------------------------------------------------------------------------
# Wikipedia index-constituent fetchers — pull current, exhaustive lists.
# Each returns dict[symbol] = name. Yahoo Finance suffixes added per exchange.
# ---------------------------------------------------------------------------

WIKIPEDIA_INDEX_SOURCES: dict[str, dict] = {
    "S&P 500 (USA)": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "symbol_col": "Symbol",
        "name_col": "Security",
        "suffix": "",
    },
    "NASDAQ-100 (USA)": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "table_index": 4,
        "symbol_col": "Ticker",
        "name_col": "Company",
        "suffix": "",
    },
    "FTSE 100 (UK)": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_index": 4,
        "symbol_col": "Ticker",
        "name_col": "Company",
        "suffix": ".L",
    },
    "DAX (Germany)": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "table_index": 4,
        "symbol_col": "Ticker",
        "name_col": "Company",
        "suffix": ".DE",
    },
    "CAC 40 (France)": {
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "table_index": 4,
        "symbol_col": "Ticker",
        "name_col": "Company",
        "suffix": ".PA",
    },
    "Hang Seng Index (HK)": {
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "table_index": 6,
        "symbol_col": "Ticker",
        "name_col": "Name",
        "suffix": ".HK",
    },
    "NIFTY 50 (India)": {
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "table_index": 2,
        "symbol_col": "Symbol",
        "name_col": "Company name",
        "suffix": ".NS",
    },
    "KOSPI (Korea, sample)": {
        "url": "https://en.wikipedia.org/wiki/KOSPI",
        "table_index": 5,
        "symbol_col": "Symbol",
        "name_col": "Company",
        "suffix": ".KS",
    },
    # Note: Nikkei 225 and full TWSE lists are not on a single Wikipedia table —
    # use the curated "Asia (ex-HK/India)" preset for Japan/Taiwan coverage.
}


# Pipe-delimited symbol-directory files published by NASDAQ Trader.
# These are the *official* listing files used by US brokers — they cover
# every common stock listed on each venue and refresh nightly.
NASDAQ_TRADER_SOURCES: dict[str, dict] = {
    "All US NASDAQ-listed (~3,500)": {
        "url": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "symbol_col": "Symbol",
        "name_col": "Security Name",
        "exclude_etf": True,
    },
    "All US NYSE / AMEX / Arca (~6,000)": {
        "url": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        # otherlisted has both "ACT Symbol" (raw) and "NASDAQ Symbol" (cleaned); use ACT.
        "symbol_col": "ACT Symbol",
        "name_col": "Security Name",
        "exclude_etf": True,
    },
}
