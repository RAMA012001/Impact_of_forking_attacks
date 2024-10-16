import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import numpy as np

# import os
# os.chdir(Path().absolute())


DAILY_DATA_DIR = "daily_data"
MIN_DATA_DIR = "min_data"


@dataclass
class MarketIndex:
    name: str
    file_name: str

    @staticmethod
    def calculate_returns(data, close_price_col):
        data["log_returns"] = np.log(
            data[close_price_col] / data[close_price_col].shift(1)
        )
        data["discrete_returns"] = (
            data[close_price_col] / data[close_price_col].shift(1) - 1
        )

    def load_data(self):
        """Load data from the CSV file and reverse the order."""
        try:
            self.data_path = Path(MIN_DATA_DIR, self.file_name)
            self.data = pd.read_csv(self.data_path)
            print(f"Data for {self.name} loaded successfully.")
        except Exception as e:
            self.data = None
            print(f"Failed to load data for {self.name}: {e}")

        self.data.set_index("time", inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        self.data.index = self.data.index.tz_localize("UTC")
        self.data = self.data.ffill()
        # Calculate returns
        self.calculate_returns(self.data, "CCMIX")


@dataclass
class Cryptocurrency(MarketIndex):
    attack_dates: list  # Non-default argument must come first

    def load_data(self):
        """Load data from the CSV file and reverse the order."""
        try:
            self.data_path = Path(DAILY_DATA_DIR, self.file_name)
            self.data = pd.read_csv(self.data_path, sep=";").iloc[::-1]
            self.data["Currency"] = self.name
            print(f"Data for {self.name} loaded successfully.")
        except Exception as e:
            self.data
            print(f"Failed to load data for {self.name}: {e}")

        # Dropping unnecessary columns
        self.data.drop(
            ["timeClose", "timeHigh", "timeLow", "open", "high", "low", "timestamp"],
            axis=1,
            inplace=True,
        )
        self.data.rename(columns={"timeOpen": "time"}, inplace=True)
        self.data.set_index("time", inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        self.calculate_returns(self.data, "close")

    def add_market_returns(self, market_data):
        """Merge market returns into the current data."""
        try:
            market_discrete_returns_col = "market_discrete_returns"
            market_log_returns_col = "market_log_returns"
            self.data = self.data.merge(
                market_data.rename(
                    columns={
                        "discrete_returns": market_discrete_returns_col,
                        "log_returns": market_log_returns_col,
                    }
                ),
                left_index=True,
                right_index=True,
            )[
                list(self.data.columns)
                + [market_discrete_returns_col, market_log_returns_col]
            ]
            print(f"Market returns added successfully to {self.name} data.")
        except Exception as e:
            print(f"Failed to add market returns for {self.name}: {e}")

    def add_bitcoin_returns(self, bitcoin_data):
        """Merge Bitcoin returns into the current data."""
        try:
            bitcoin_discrete_returns_col = "bitcoin_discrete_returns"
            bitcoin_log_returns_col = "bitcoin_log_returns"
            self.data = self.data.merge(
                bitcoin_data.rename(
                    columns={
                        "discrete_returns": bitcoin_discrete_returns_col,
                        "log_returns": bitcoin_log_returns_col,
                    }
                ),
                left_index=True,
                right_index=True,
            )[
                list(self.data.columns)
                + [bitcoin_discrete_returns_col, bitcoin_log_returns_col]
            ]
            print(f"Bitcoin returns added successfully to {self.name} data.")
        except Exception as e:
            print(f"Failed to add Bitcoin returns for {self.name}: {e}")


CCMIX = MarketIndex(name="CCMIX", file_name="10min_CCMIX.csv")


# Load data for each cryptocurrency
class Cryptos(Enum):
    BTG = Cryptocurrency(
        name="Bitcoin Gold",
        attack_dates=["2018-05-16 22:37:54", "2020-01-23 18:01:32"],
        file_name=(
            "Bitcoin Gold_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv"
        ),
    )

    EMC2 = Cryptocurrency(
        name="Einsteinium",
        attack_dates=["2019-01-06 12:00:00"],
        file_name="Einsteinium_14_10_2017-13_12_2017_historical_data_coinmarketcap.csv",
    )

    ELC = Cryptocurrency(
        name="Electroneum",
        attack_dates=["2018-04-04 12:00:00"],
        file_name="Electroneum_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv",
    )

    ETC = Cryptocurrency(
        name="Ethereum Classic",
        attack_dates=[
            "2019-01-04 03:27:11",
            "2020-07-31 16:36:07",
            "2020-08-06 02:54:27",
            "2020-08-29 00:00:00",
        ],
        file_name=(
            "Ethereum Classic_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv"
        ),
    )

    EXP = Cryptocurrency(
        name="Expanse",
        attack_dates=["2019-07-29 12:00:00"],
        file_name="Expanse_14_10_2017-13_12_2017_historical_data_coinmarketcap.csv",
    )

    # FTC = Cryptocurrency(
    #     name='Feathercoin',
    #     attack_dates=[],  # Add appropriate attack dates if available
    #     file_name='Feathercoin_05_04_2013-04_06_2013_historical_data_coinmarketcap.csv'
    # )

    XZC = Cryptocurrency(
        name="Firo",
        attack_dates=["2021-01-19 17:24:20"],
        file_name="Firo_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv",
    )

    ZEN = Cryptocurrency(
        name="Horizen",
        attack_dates=["2018-06-03 00:26:00"],
        file_name="Horizen_14_10_2017-13_12_2017_historical_data_coinmarketcap.csv",
    )

    KRB = Cryptocurrency(
        name="Karbo",
        attack_dates=["2018-10-11 12:00:00"],
        file_name="Karbo_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv",
    )

    LTC = Cryptocurrency(
        name="Litecoin",
        attack_dates=["2018-05-30 12:00:00", "2019-06-04 12:00:00"],
        file_name="Litecoin_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv",
    )

    MNC = Cryptocurrency(
        name="MonaCoin",
        attack_dates=["2018-04-08 12:00:00", "2018-05-15 12:00:00"],
        file_name="MonaCoin_14_10_2017-13_12_2017_historical_data_coinmarketcap.csv",
    )

    # PGC = Cryptocurrency(
    #     name='Pigeoncoin',
    #     attack_dates=[],  # Add appropriate attack dates if available
    #     file_name='Pigeoncoin_10_06_2018-09_08_2018_historical_data_coinmarketcap.csv'
    # )

    PIN = Cryptocurrency(
        name="Public Index Network",
        attack_dates=["2018-09-08 12:00:00"],
        file_name=(
            "Public Index Network_13_12_2017-12_02_2018_historical_data_"
            "coinmarketcap.csv"
        ),
    )

    # TRC = Cryptocurrency(
    #     name='Terracoin',
    #     attack_dates=[],  # Add appropriate attack dates if available
    #     file_name='Terracoin_30_03_2013-29_05_2013_historical_data_coinmarketcap.csv'
    # )

    XVG = Cryptocurrency(
        name="Verge",
        attack_dates=["2018-04-04 06:00:00", "2018-05-22 00:37:00"],
        file_name="Verge_13_12_2017-12_02_2018_historical_data_coinmarketcap.csv",
    )

    # VTC = Cryptocurrency(
    #     name='Vertcoin',
    #     attack_dates=[],  # Add appropriate attack dates if available
    #     file_name='Vertcoin_01_01_2018-31_12_2021_historical_data_coinmarketcap.csv'
    # )
