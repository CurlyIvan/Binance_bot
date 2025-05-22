#!/usr/bin/env python3
"""Binance Three-Layer Momentum Bot – main.py (v2.2)
====================================================
* **Layer 3** – ML regime gate (LightGBM + calibrated LogisticRegression ensemble).
* **Layer 1** – daily dual-momentum rotator (top-5 USDT pairs).
* **Layer 2** – 15-minute breakout turbo add-on with trailing stop (skipped on ML-red days).
* Retrains ML at every launch **and** monthly (1st @ 03:00 UTC).
* CSV logging for trades, balance, wallet; support `--suffix` for A/B tests.

Usage:
```bash
python main.py --paper             # paper mode (no live orders)
python main.py --paper --suffix=_B # parallel variant
```"""
from __future__ import annotations
import argparse, csv, json, logging, math, os, pathlib, sys, time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib, numpy as np, pandas as pd
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from binance.enums import HistoricalKlinesType
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Optional LightGBM
try:
    import lightgbm as lgb
    USE_LGB = True
except Exception:
    USE_LGB = False

# Strategy params
TOP_N, HOLD_N = 30, 5
ABS_MOM_DAYS, EMA_LONG = 30, 200
BREAKOUT_TF, VOL_MUL, TRAIL_ATR_MUL, ATR_LEN_15 = "15m", 1.5, 1.0, 15
TARGET_ANNUAL_VOL = 0.20
DAILY_REBALANCE_UTC = 2
THRESH_PROB = 0.60
FEATURES = [
    "ret5_prev", "atr_pct_prev", "funding_d3_prev", "oi_ratio_curr", "mom15m",
    "atr_pct_15m", "ethbtc_mom", "dxy_change",
]
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"

# CLI args
parser = argparse.ArgumentParser(description="Binance 3-layer momentum bot")
parser.add_argument("--paper",   action="store_true", help="paper mode (no live orders)")
parser.add_argument("--retrain", action="store_true", help="retrain model & exit")
parser.add_argument("--suffix",  default="",           help="state-folder suffix for A/B tests")
args = parser.parse_args()
MODE = "paper" if args.paper else "live"
SUFFIX = args.suffix

# Paths & env
load_dotenv()
ROOT       = pathlib.Path(__file__).resolve()
DATA_DIR   = ROOT.parent / f"{ROOT.stem}_state{SUFFIX}"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE   = DATA_DIR / "bot.log"
POS_FILE   = DATA_DIR / "positions.json"
MODEL_FILE = DATA_DIR / "ml_filter.pkl"
TRADES_CSV = DATA_DIR / "trades.csv"
BAL_CSV    = DATA_DIR / "balance.csv"
WALLET_CSV = DATA_DIR / "wallet.csv"

# Logging
logging.basicConfig(level=logging.INFO,
                    format=LOG_FMT,
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_FILE, encoding="utf-8")])
logger = logging.getLogger("bot")
if MODE == "paper":
    logger.info("PAPER mode – no live orders will be sent")

# CSV init
for path, header in (
    (TRADES_CSV, ["ts","symbol","side","qty","price","value_usdt","pnl_usdt"]),
    (BAL_CSV,    ["date","equity_usdt","ml_prob","gate"]),
    (WALLET_CSV, ["date","assets_json"]),
):
    if not path.exists():
        path.write_text(",".join(header)+"\n")

# Binance client (always available for public data)
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)
# set mode based on credentials
if API_KEY and API_SECRET:
    MODE = "live"
else:
    MODE = "paper"
logger = logging.getLogger("bot")
if MODE == "paper":
    logger.info("PAPER mode – no live orders will be sent")

# CSV loggers
def log_trade(ts,sym,side,qty,price,pnl=0.0):
    with TRADES_CSV.open('a', newline='') as f:
        csv.writer(f).writerow([ts,sym,side,round(qty,6),round(price,6),round(qty*price,2),round(pnl,2)])

def log_balance(eq,prob):
    gate = "green" if prob>=THRESH_PROB else "red"
    with BAL_CSV.open('a', newline='') as f:
        csv.writer(f).writerow([datetime.utcnow().date(),round(eq,2),round(prob,4),gate])

def log_wallet():
    if MODE=='paper':
        assets={'USDT':10000.0}
    else:
        info=client.get_account()
        assets={b['asset']:float(b['free']) for b in info['balances'] if float(b['free'])>0}
    with WALLET_CSV.open('a', newline='') as f:
        csv.writer(f).writerow([datetime.utcnow().date(),json.dumps(assets,separators=(',',':'))])

# Market data
def get_klines(sym,interval,lookback)->pd.DataFrame:
    raw=client.get_historical_klines(sym,interval,lookback,klines_type=HistoricalKlinesType.SPOT)
    df=pd.DataFrame(raw,columns=["open_time","o","h","l","c","v","x","y","z","u","w","q"],dtype=float)
    df.open_time=pd.to_datetime(df.open_time,unit='ms',utc=True)
    df.set_index('open_time',inplace=True)
    df.rename(columns={"c":"close","h":"high","l":"low","v":"v"},inplace=True)
    return df[['close','high','low','v']]

def stooq_pct(code):
    try:
        df=pd.read_csv(f"https://stooq.com/q/d/l/?s={code}&i=d")
        return (df.Close.iloc[-1]-df.Close.iloc[-2])/df.Close.iloc[-2]
    except: return 0.0

def funding_series(days):
    fr=client.futures_funding_rate(symbol='BTCUSDT', limit=min(days*3,1000))
    vals=[float(x['fundingRate']) for x in fr]
    return pd.Series(vals)[::-1].groupby(np.arange(len(vals))[::-1]//3).mean()[-days:]

def funding_d3():
    s=funding_series(4)
    return s.iloc[-1]-s.iloc[-4] if len(s)==4 else 0.0

def oi_ratio():
    hist=client.futures_open_interest_hist(symbol='BTCUSDT', period='1d', limit=30)
    ser=pd.Series([float(x['sumOpenInterest']) for x in hist])
    return ser.iloc[-1]/ser[:-1].mean() if ser[:-1].mean() else 1.0

def universe_top_volume():
    t=client.get_ticker()
    u=[x for x in t if x['symbol'].endswith('USDT')]
    return [x['symbol'] for x in sorted(u,key=lambda y:float(y['quoteVolume']),reverse=True)[:TOP_N]]

# ML layer
LAST_ML_PROB:float|None=None
class EnsembleModel(BaseEstimator):
    def __init__(self,m1,m2): self.m1, self.m2 = m1, m2
    def predict_proba(self,X):
        p1=self.m1.predict_proba(X)[:,1]
        p2=self.m2.predict_proba(X)[:,1]
        avg=(p1+p2)/2
        return np.vstack([1-avg,avg]).T

def build_feature_vector()->pd.DataFrame:
    btc=get_klines('BTCUSDT','1d','41 day ago UTC')
    df15=get_klines('BTCUSDT','15m','2 day ago UTC')
    ethbtc=get_klines('ETHBTC','1d','41 day ago UTC')
    fv={
        'ret5_prev':btc.close.pct_change(5).shift(1).iloc[-1],
        'atr_pct_prev':AverageTrueRange(btc.high,btc.low,btc.close,5).average_true_range().shift(1).iloc[-1]/btc.close.shift(1).iloc[-1],
        'funding_d3_prev':funding_d3(),
        'oi_ratio_curr':oi_ratio(),
        'mom15m':df15.close.pct_change().iloc[-1],
        'atr_pct_15m':AverageTrueRange(df15.high,df15.low,df15.close,96).average_true_range().iloc[-1]/df15.close.iloc[-1],
        'ethbtc_mom':ethbtc.close.pct_change(5).iloc[-1],
        'dxy_change':stooq_pct('USDOLLAR'),
    }
    return pd.DataFrame([fv]).fillna(0)

def retrain_model():
    logger.info('Retraining ML model …')
    btc=get_klines('BTCUSDT','1d','1100 day ago UTC').tail(1095)
    btc['ret5']=btc.close.pct_change(5)
    btc['ret5_prev']=btc.ret5.shift(1)
    btc['atr_pct']=AverageTrueRange(btc.high,btc.low,btc.close,5).average_true_range()/btc.close
    btc['atr_pct_prev']=btc.atr_pct.shift(1)
    btc['funding_d3_prev']=funding_d3()
    btc['oi_ratio_curr']=oi_ratio()
    df15_all=get_klines('BTCUSDT','15m','1100 day ago UTC')
    mom15m_daily=df15_all.close.pct_change().resample('1D').last().reindex(btc.index).fillna(0)
    btc['mom15m']=mom15m_daily.values
    atr15_series=AverageTrueRange(df15_all.high,df15_all.low,df15_all.close,96).average_true_range()/df15_all.close
    btc['atr_pct_15m']=atr15_series.resample('1D').last().reindex(btc.index).ffill()
    ethbtc=get_klines('ETHBTC','1d','1100 day ago UTC').tail(len(btc))
    btc['ethbtc_mom']=ethbtc.close.pct_change(5).values
    btc['dxy_change']=stooq_pct('USDOLLAR')
    btc.fillna(0,inplace=True)
    btc['label']=(btc.ret5.abs()>1.5*btc.atr_pct).astype(int)
    X,y=btc[FEATURES],btc.label
    split=-365
    if USE_LGB:
        gb=lgb.LGBMClassifier(objective='binary',learning_rate=0.05,num_leaves=31,feature_fraction=0.8,n_estimators=300,min_gain_to_split=0.001,verbose=-1)
        gb.fit(X[:split],y[:split])
        gbc=CalibratedClassifierCV(gb,cv='prefit',method='sigmoid').fit(X[split:],y[split:])
    else:
        gbc=None
    lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X[:split],y[:split])
    lrc=CalibratedClassifierCV(lr,cv='prefit',method='sigmoid').fit(X[split:],y[split:])
    model=EnsembleModel(gbc,lrc) if gbc else lrc
    auc=roc_auc_score(y[split:],model.predict_proba(X[split:])[:,1])
    joblib.dump(model,MODEL_FILE)
    logger.info(f'Model saved – AUC {auc:.3f}')

def ml_filter_allows_trading()->bool:
    global LAST_ML_PROB
    if not MODEL_FILE.exists():LAST_ML_PROB=1.0;return True
    model=joblib.load(MODEL_FILE)
    LAST_ML_PROB=float(model.predict_proba(build_feature_vector())[:,1][0])
    logger.info(f'ML probability = {LAST_ML_PROB:.4f}')
    return LAST_ML_PROB>=THRESH_PROB

# Trading state
positions:Dict[str,Dict]=json.loads(POS_FILE.read_text()) if POS_FILE.exists() else {}

def save_positions():
    POS_FILE.write_text(json.dumps(positions,indent=2))

# Layer 1

def dual_momentum(sym)->Tuple[float,bool]:
    df=get_klines(sym,'1d',f'{ABS_MOM_DAYS+EMA_LONG} day ago UTC')
    ema=EMAIndicator(df.close,EMA_LONG).ema_indicator().iloc[-1]
    rel=df.close.pct_change(ABS_MOM_DAYS).iloc[-1]
    return rel, (df.close.iloc[-1]>ema and rel>0)

def vol_target_qty(sym,equity)->float:
    price=float(client.get_symbol_ticker(symbol=sym)['price'])
    df=get_klines(sym,'1h','14 day ago UTC')
    atr=AverageTrueRange(df.high,df.low,df.close,14).average_true_range().iloc[-1]
    atr_pct=atr/price
    usd_tgt=(equity*TARGET_ANNUAL_VOL)/(atr_pct*math.sqrt(252)) if atr_pct else 0
    return round(usd_tgt/price,6)

def open_position(sym):
    equity=float(client.get_account()['totalWalletBalance']) if MODE=='live' else 10000.0
    qty=vol_target_qty(sym,equity)
    if qty<=0: return
    logger.info(f"{'PAPER' if MODE=='paper' else 'LIVE'} BUY {qty:.6f} {sym}")
    if MODE=='live': client.order_market_buy(symbol=sym,quantity=qty)
    price=float(client.get_symbol_ticker(symbol=sym)['price'])
    positions[sym]={'entry':price,'quantity':qty,'breakout':False}
    save_positions()

def close_position(sym):
    if sym not in positions: return
    qty=positions[sym]['quantity']
    logger.info(f"{'PAPER' if MODE=='paper' else 'LIVE'} SELL {qty:.6f} {sym}")
    if MODE=='live': client.order_market_sell(symbol=sym,quantity=qty)
    positions.pop(sym,None)
    save_positions()

def daily_rebalance():
    logger.info('=== Daily Rebalance ===')
    if not ml_filter_allows_trading():
        logger.info('ML red – flattening positions')
        for s in list(positions): close_position(s)
        return
    winners=[]
    for s in universe_top_volume():
        r,ok=dual_momentum(s)
        if ok: winners.append((s,r))
    leaders=[s for s,_ in sorted(winners,key=lambda x:x[1],reverse=True)[:HOLD_N]]
    for s in list(positions):
        if s not in leaders: close_position(s)
    for s in leaders:
        if s not in positions: open_position(s)
    logger.info(f"Holding: {list(positions)}")

# Layer 2

def monitor_breakouts():
    if not ml_filter_allows_trading() or not positions: return
    for sym,pos in list(positions.items()):
        try:
            y_high=get_klines(sym,'1d','2 day ago UTC').high.iloc[-2]
            df15=get_klines(sym,BREAKOUT_TF,'1 day ago UTC')
            last=df15.iloc[-1]; vol_avg=df15.v.tail(20).mean()
            if not pos['breakout'] and last.close>y_high and last.v>VOL_MUL*vol_avg:
                atr15=AverageTrueRange(df15.high,df15.low,df15.close,ATR_LEN_15).average_true_range().iloc[-1]
                qty_add=round(pos['quantity']*0.5,6)
                logger.info(f"Turbo BUY {qty_add:.6f} {sym}")
                if MODE=='live': client.order_market_buy(symbol=sym,quantity=qty_add)
                pos['quantity']+=qty_add; pos['breakout']=True
                pos['stop']=last.close-TRAIL_ATR_MUL*atr15; save_positions()
            if 'stop' in pos:
                new_stop=last.close-TRAIL_ATR_MUL*atr15
                if new_stop>pos['stop']: pos['stop']=new_stop
                if last.close<pos['stop']: close_position(sym)
        except Exception as e:
            logger.error(f"Breakout error {sym}: {e}")

# Daily summary

def daily_summary():
    """
    Log today's balance, ML gate, and wallet snapshot; handle permission errors gracefully.
    """
    # fetch equity safely
    try:
        if MODE == 'live' and client:
            acct = client.get_account()
            equity = float(acct.get('totalWalletBalance', 0.0))
        else:
            equity = 10000.0
    except Exception as e:
        logger.warning(f"Account fetch failed ({e}), using fallback equity")
        equity = 10000.0

    # log balance and ML probability
    prob = LAST_ML_PROB if LAST_ML_PROB is not None else 1.0
    log_balance(equity, prob)

    # fetch wallet safely
    try:
        if MODE == 'live' and client:
            acct = client.get_account()
            assets = {b['asset']: float(b['free']) for b in acct.get('balances', []) if float(b.get('free', 0)) > 0}
        else:
            assets = {'USDT': equity}
    except Exception as e:
        logger.warning(f"Wallet fetch failed ({e}), logging fallback assets")
        assets = {'USDT': equity}
    with WALLET_CSV.open('a', newline='') as f:
        csv.writer(f).writerow([datetime.utcnow().date(), json.dumps(assets, separators=(',', ':'))])

# Scheduler

def build_scheduler()->BackgroundScheduler:
    sched=BackgroundScheduler(executors={'default':ThreadPoolExecutor(max_workers=10)},timezone=timezone.utc)
    sched.add_job(daily_rebalance,'cron',hour=DAILY_REBALANCE_UTC,minute=0,id='rebalance',misfire_grace_time=60)
    sched.add_job(monitor_breakouts,'interval',minutes=15,id='breakout',max_instances=1,coalesce=True)
    sched.add_job(retrain_model,'cron',day=1,hour=3,minute=0,id='retrain',misfire_grace_time=3600)
    return sched

# Main
if __name__=='__main__':
    retrain_model()
    if args.retrain: sys.exit(0)
    ml_filter_allows_trading()
    daily_summary()
    sched=build_scheduler(); sched.start()
    logger.info('Scheduler started – bot running (Ctrl+C to stop)')
    try:
        while True: time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown(); logger.info('Graceful shutdown')
