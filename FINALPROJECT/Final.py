import numpy as np
seed = 42
np.random.seed(seed)
import math
from scipy.stats import norm
import pandas as pd
from pandas.tseries.offsets import BDay
from FinDates.daycount import yearfrac

depo_converter = lambda x: float(x)/100
future_converter = lambda x: float(x)/100                    
swap_converter = lambda x: float(x)/100                    


DC_CONV = {"DEPO": "ACT/360" # "DEPO": "ACT/360" (given)
    , "FUTURE": "ACT/360" # "FUTURE": "30E/360" (given)
    , "SWAP" : "30E/360" # "SWAP" : "30E/360" (given)
    , "BOND" : "30E/360" # Macaulay duration
    , "INTERP" : "ACT/365" # USED for delta(t_0, t_i) to get the interpolated zero rates
    , "PARTY_A" : "ACT/360" # "PARTY_A" : "ACT/360" (given)
    , "PARTY_B" : "30E/360" # "PARTY_B" : "30E/360" (given)
    }


def getZeroRates(dates, df):
    # dates : yearfractions(t0,ti) list
    # df : discount factor B(t0,ti) list
    # returns list of zero rates on ti
    
    effDates, effDf = dates, df
    zeroRates = []
    
    for i in range(len(effDates)):
        zero_rate = - np.log(effDf[i]) / (effDates[i])
        zeroRates.append(zero_rate)
    
    return zeroRates


def getRatesLinInterpDiscount(dtSettle, dtRef, xDates, xDf, daycount=DC_CONV["INTERP"]):
    
    # xDates, xDf : available set of dates and discounts on which interpolate
    # dfRef: target day
    # returns discount on dtRef date
    
    assert(len(xDates) == len(xDf))
    
    # compute relevant yearfractions for available set of dates
    yearFractions = [yearfrac(dtSettle, xDates[i], daycount) for i in range(len(xDates))]
    
    # convert discounts into zero rates
    zeroRates = getZeroRates(yearFractions, xDf)
    
    # apply the interpolation on the target day
    zeroR = np.interp(yearfrac(dtSettle, dtRef, daycount), yearFractions, zeroRates)
    
    # convert zero rate into discount
    discount = np.exp(-zeroR * yearfrac(dtSettle, dtRef, daycount))
    
    return discount


def interpZeroRate(dtSettle, dtRef, xDates, xDf, daycount=DC_CONV["INTERP"]):
    
    """
    This function interpolates the zero rate on the target day using the available set of dates and discounts.
    """
    
    assert(len(xDates) == len(xDf))
    
    # compute relevant yearfractions for available set of dates
    yearFractions = [yearfrac(dtSettle, xDates[i], daycount) for i in range(len(xDates))]
    
    # convert discounts into zero rates
    zeroRates = getZeroRates(yearFractions, xDf)
    
    # apply the interpolation on the target day
    zeroR = np.interp(yearfrac(dtSettle, dtRef, daycount), yearFractions, zeroRates)
    
    return zeroR


def bootstrapDepo(dtSettle, df_depo, df_futures, termDates, discounts, keepDepos = 3, shift = 0.0):
    
    df_depo = df_depo[:keepDepos]

    rate = (df_depo["ASK"] + df_depo["BID"]) / 2 + shift# L(t_0, t_i), # shift is added when needed

    depoYearFrac = [] # delta(t_0, t_i)
    for i in range(df_depo.index.shape[0]):
        depoYearFrac.append(yearfrac(dtSettle, df_depo.index[i], DC_CONV['DEPO']))

    for i in range(df_depo.index.shape[0]):
        termDates.append(df_depo.index[i])
        discounts.append(1/(1 +depoYearFrac[i] * rate[i]))
    
    return termDates, discounts


def bootstrapFuture(dtSettle, df_futures, termDates, discounts, keepFutures = 7, shift = 0.0):
    
    df_futures = df_futures[:keepFutures]

    rate = 1 - (df_futures["ASK"] + df_futures["BID"]) / 2 + shift # L(t_0; t_{i-1}, t_i); approximation; shift is added when needed

    t1 = df_futures["Settle"]
    t2 = df_futures["Expiry"]

    futureYearFrac = [] # delta(t_{i-1}, t_i)
    for i in range(df_futures.index.shape[0]):
        futureYearFrac.append(yearfrac(t1[i], t2[i], DC_CONV['FUTURE'])) # proper convention is used

    futureFwdDiscounts = [] # B(t0;t_{i-1},t_i) = 1/(1 + L(t0;t_{i-1},t_i) * delta(t_{i-1}, t_i))
    for i in range(df_futures.index.shape[0]):
        futureFwdDiscounts.append(1/(1 + futureYearFrac[i] * rate[i]))

    # B(t_0, t_i) = B(t_0;t_{i-1},t_{i}) * B(t_0, t_{i-1})
    for i in range(df_futures.index.shape[0]):
        if i == 0: # no need for interpolation
            termDates.append(df_futures['Expiry'][i])
            discounts.append(futureFwdDiscounts[i]*discounts[-1])
        else: # interpolation needed for B(t_0, t_{i-1 })
            """
            # NUMPY INTERPOLATION
            
            termDatesarray = np.zeros(len(termDates))
            for j in range(len(termDates)):
                termDatesarray[j] = termDates[j].to_numpy()
            interp = np.interp(df_futures['Settle'][i].to_numpy(), termDatesarray, discounts)
            discounts.append(futureFwdDiscounts[i]*interp)
            """
            discounts.append(futureFwdDiscounts[i]*getRatesLinInterpDiscount(dtSettle, df_futures['Settle'][i], termDates, discounts))
            termDates.append(df_futures['Expiry'][i])
    
    return termDates, discounts


def bootstrapSwap(dtSettle, df_swaps, termDates, discounts, shift = 0.0):
    
    rate = (df_swaps["ASK"] + df_swaps["BID"])/2 + shift # shift is added when needed

    swapYearFrac = [] # delta(t_{i-1}, t_i)
    for i in range(df_swaps.index.shape[0]): 
        if i == 0:
            swapYearFrac.append(yearfrac(dtSettle, df_swaps.index[i], DC_CONV['SWAP']))
        else:
            swapYearFrac.append(yearfrac(df_swaps.index[i-1], df_swaps.index[i], DC_CONV['SWAP']))

    """
    # NUMPY INTERPOLATION

    termDatesarray = np.zeros(len(termDates))
    for i in range(len(termDates)):
        termDatesarray[i] = termDates[i].to_numpy()

    initial_int_disc = np.interp(df_swaps.index[0].to_numpy(), 
                                 termDatesarray, 
                                 discounts)
    """
    
    initialIntDisc = getRatesLinInterpDiscount(dtSettle, df_swaps.index[0], termDates, discounts)

    BPV = swapYearFrac[0] * initialIntDisc # initial BPV, with interpolated discount
    S = rate 

    for i in range(1, df_swaps.index.shape[0]):
        termDates.append(df_swaps.index[i])
        discounts.append((1 - S[i] * BPV) / (1 + S[i] * swapYearFrac[i]))
        BPV += swapYearFrac[i] * discounts[-1]
        
    return termDates, discounts


def bootstrapCurves(dtSettle = pd.to_datetime("2008-02-19"), # settlement date
                    df_depo = pd.DataFrame(), 
                    df_futures = pd.DataFrame(), 
                    df_swaps = pd.DataFrame(), 
                    shift = 0.0):
    
    """
    This function bootstraps the discount curve and zero curve from the given market data.
    RMK: works with bps shifts as well
    """
    
    dates = [dtSettle]  
    discounts = [1.] 

    dates, discounts = bootstrapDepo(dtSettle, df_depo, df_futures, dates, discounts, shift=shift)

    dates, discounts = bootstrapFuture(dtSettle, df_futures, dates, discounts, shift=shift)

    dates, discounts = bootstrapSwap(dtSettle, df_swaps, dates, discounts, shift=shift)


    yearFrac = [yearfrac(dtSettle, T, DC_CONV["INTERP"]) for T in dates[1:]]
    zeroRates_shift = getZeroRates(yearFrac, discounts[1:])

    discCurve = pd.Series(index=pd.to_datetime(dates), data=discounts)
    zeroCurve = pd.Series(index=pd.to_datetime(dates[1:]), data=zeroRates_shift)
    
    return discCurve, zeroCurve, dates, discounts


def followingBday(day): 
    """
    # day : datetime object
    # returns the following business day if the given day is not a business day
    """
    
    adjusted_day = day
    while adjusted_day.weekday() > 4:  # While adjusted_day is a weekend (Saturday or Sunday)
        adjusted_day += BDay(1)        # Move to the next business day
        
    return adjusted_day


def callEuropean(S: float, K: float, T: float, r: float, d: float, sigma: float):
    """
    This function copute the price of a European call option using the Garman-Kohlhagen model.
    :param F: initial underlying price
    :param K: strike price
    :param T: time to maturity
    :param r: risk-free interest rate
    :param d: dividend yield
    :param sigma: volatility
    """

    d1 = (math.log(S / K) + (r - d + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    call_price = S*np.exp(-d*T)*N_d1 - K*np.exp(-r*T)*N_d2 
    
    return call_price


def deltaEuropean(S, K, T, r, d, sigma, optiontype=0):
    """
    Compute the Delta of a plain vanilla european option on a forward, 
    the first order price sensitivity wrt the underlying asset price.
    S: underlying price t0
    K: option strike
    T: time to maturity
    r: risk free rate
    d: dividend yield
    sigma: volatility 
    optiontype: flag to specify if the option is a call (0) or a put (1)

    """
    
    d1 = (math.log(S / K) + (r - d + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))
    
    delta = norm.cdf(d1) - optiontype

    return delta * np.exp(-d*T)

def vegaEuropean(S, K, T, r, d, sigma):
    """
    Compute the Vega of a european option on a forward,
    the first order price sensitivity wrt the volatility.
    S: underlying price
    K: option strike
    T: time to maturity
    r: risk free rate
    d: dividend yield
    """
    
    d1 = (math.log(S/ K) + (r - d + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))

    vega = S* np.exp(-d*T) * math.sqrt(T) * norm.pdf(d1) 

    return vega


