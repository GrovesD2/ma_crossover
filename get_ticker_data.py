from utils import tickers
     
if __name__ == "__main__":
    
    # Switch for the download type,
    # 'user' downloads all the tickers in "tickers.csv"
    # 'spy' downloads all the current s&p 500 tickers
    download_type = 'user'
    
    # Check the above option is correct and set the file-paths
    tickers.check_run(download_type)
    
    # Grab the list of tickers and run the downloader
    tickers.get_data(tickers.get_tickers(download_type))