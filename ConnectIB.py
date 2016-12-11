# connectIB.py
"""
Use IBPy to interact b/w Python and IB TWS.
IBPy is a wrapper of the naive Java API of IB TWS.

Created on Tue Nov 01 23:10:21 2016
@author: Linchang
"""

from ib.opt import Connection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order

# IB provides us with the capability of handling errors and server responses by 
# a callback mechanism. The following two functions do nothing more than print 
# out the contents of the messages returned from the server.
def error_handler(msg):
    """Handles the capturing of error messages"""
    print "Server Error: %s" % msg

def reply_handler(msg):
    """Handles of server replies"""
    print "Server Response: %s, %s" % (msg.typeName, msg)

# The following two functions wrap the creation of the Contract and Order objects
def create_contract(symbol, sec_type, exch, prim_exch, curr):
    """Create a Contract object defining what will
    be purchased, at which exchange and in which currency.

    symbol - The ticker symbol for the contract
    sec_type - The security type for the contract ('STK' is 'stock')
    exch - The exchange to carry out the contract on
    prim_exch - The primary exchange to carry out the contract on
    curr - The currency in which to purchase the contract."""
    
    contract = Contract()
    contract.m_symbol = symbol
    contract.m_secType = sec_type
    contract.m_exchange = exch
    contract.m_primaryExch = prim_exch
    contract.m_currency = curr
    return contract

def create_order(order_type, quantity, action, price=None):
    """Create an Order object (Market/Limit) to go long/short.

    order_type - 'MKT', 'LMT' for Market or Limit orders
    quantity - Integral number of assets to order
    action - 'BUY' or 'SELL'."""
    
    if (price is not None and order_type=='LMT'):
        order = Order()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action
        order.m_lmtPrice = price
        
    elif (price is None and order_type=='MKT'):
        order = Order()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action
        
    else: 
        raise Exception("Specify correct order type and price!")
    
    return order

def main():
    # Connect to the Trader Workstation (TWS) running on the
    # usual port of 7496 (mine is 7497), with a clientId of 999
    # (The clientId is chosen by us and we will need 
    # separate IDs for both the execution connection and
    # market data connection)
    tws_conn = Connection.create(port=7496, clientId=999)
    tws_conn.connect()

    # Assign the error handling function defined above to the TWS connection
    tws_conn.register(error_handler, 'Error')
    tws_conn.registerAll(reply_handler)
    
    # Create an order ID which is 'global' for this session.
    # In a production system this must be incremented for each trade order.
    order_id = 124
    
    # Create a contract in GOOG stock via SMART order routing
    goog_contract = create_contract('GOOG', 'STK', 'SMART', 'SMART', 'USD')
    # Go long 100 shares of Google
    goog_order = create_order('MKT', 100, 'BUY')
    # Use the connection to the send the order to IB
    tws_conn.placeOrder(order_id, goog_contract, goog_order)
    
    tws_conn.disconnect()
    
if __name__ == "__main__":
    main()
    