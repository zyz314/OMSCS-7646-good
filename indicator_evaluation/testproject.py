import TheoreticallyOptimalStrategy as tos  
import datetime as dt
import indicators

def author():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """
    return "mshihab6"  # Change this to your user ID

def main():
    start_date = dt.datetime(2008, 1, 1)
    end_date   = dt.datetime(2009,12,31)
    symbol = "JPM"
    df_trades = tos.testPolicy(symbol = symbol, 
                               sd=start_date,
                               ed=end_date,
                               sv = 100_000) 
    indicators.run(symbols=symbol, start_date=start_date, end_date=end_date)    

if __name__=="__main__":
    main()