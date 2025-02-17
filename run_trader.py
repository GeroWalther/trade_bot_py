from lumibot.traders import Trader
from trading_state import initialize_strategy
from config import OANDA_CREDS

def main():
    strategy = initialize_strategy(OANDA_CREDS)
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

if __name__ == "__main__":
    main() 