// Min and max dates in this dataset
select min date, max date from trades
// 2010.05.05    2016.04.21

// Actually, 2010.05.05 and 2010.05.06 are outside the normal range, and were
// added to the dataset at my request. Let's correct for this...
select min date, max date from trades where date > 2010.05.06
// 2013.04.21    2016.04.21

// What ES (S&P 500 E-Mini) syms were traded on 2016.04.21
select distinct sym from trades where date=2016.04.21, sym like "ES*"
// ESM16, ESU16, ESZ16

// How many trades in each on that date?
select count i by sym from trades where date=2016.04.21, sym like "ES*"
// ESM16: 181223, ESU16: 417, ESZ16L 47

// Let's look at the number of trades per minute in ESM16 on 2016.04.21
select count i by 60000 xbar time from trades where date=2016.04.21, sym=`ESM16
// Looks like trading kicks in around 08:30 and calms down by 15:00

// What's the most actively traded S&P 500 E-Mini future on a particular date?
aaa: select sym:first sym where tradecount = max tradecount, tradecount:first tradecount where tradecount = max tradecount by date from (
select tradecount:count i by date, sym from trades where date within 2013.04.21 2016.04.21, sym like "ES*"
)

// Now saw together the daily "closes" for the most liquid futures
select date, price from (
() ,/ {select last date, last time, last sym, last price from trades where date=x[`date], sym=x[`sym], time < 15:00:00.000} each () xkey aaa
) where not null date

// Now saw together the hourly price time series
select date+time, price from
() ,/ {() xkey select last date, last time, last sym, last price by (6*60*60000) xbar time from trades where date=x[`date], sym=x[`sym]} each () xkey aaa
