## The Notebook for this project is: ./PracticalApplicationII.ipynb

### Introduction

The purpose of this project is to use the CRISP-DM method to identify the characteristics of used car sales which optimize price. The following is a transcription of the results, which may be found with additional images in the Jupyter notebook shown above.

### Report

To whom it may concern, 

The following conclusions are made using a dataset of nearly half a million car sales. The dataset is initially cleaned its outliers pruned.

The average used car price comes out to roughly 17k, for a total market volume of 5.8 billion.

#### General results: low- and high-end markets

Following our analysis, I have concluded that the used car market is composed of roughly two large groups which encapsulate nearly all revenue in used car sales: low-end and the high-end vehicle markets. The primary means by which this analysis was conducted was by performing a clustering analysis -- essentially 'clustering' members into each of those groups as neatly as possible.

We find that the low-end market has a total value of 1.78 billion, with an average car price of roughly 13k, and encapsulates about 40% of all car sales.

The higher-end market has a total value of 3.53 billion and an average car price of 25k, encapsulating 42% of all car sales.

While both cover a relatively similar number of purchases, the total market value of the higher-end market is twice as great as that of the lower-end market.

The high-end market has a broader distribution of prices, which suggests that the prices themselves have more wiggle-room. The vast majority of sales are closed in the range of 17 - 40k. There is some issues with outliers in the low-end of this cluster, however those are vehicles which were likely transferred for free, such as between family members.

The lower-end market has much less variance and is sharply peaked around its average value. We can expect much less wiggle-room when it comes to price in this segment.

#### Characteristics of the low-end market

We can use simple correlation analysis to provide characteristics in a used vehicle that different segments value more or less.

In the lower-end market, customers: 

Tend to value (in order of importance)
- Transmission: automatic
- Condition: excellent
- Fuel: gas
- Year being higher (car is newer)

Tend not to value
- Transmission: other
- Condition: good
- Fuel: other
- Cylinders: 4

As such, an ideal vehicle in the low-end market will be a gas-powered, automatic transmission in good condition. They tend to value a newer car. Notably, the number of miles on the car has less significance.


#### Characteristics of the high-end market

Similarly, for the higher-end market, customers:

Tend to value
- Transmission: other
- Year being higher
- Condition: good
- Fuel: other

Tend not to value
- Transmission: automatic
- Odometer being higher
- Condition: excellent
- Size: full-size
- Fuel: gas

These characteristics are a little more varied, likely because the distribution of the customers is a little broader (see the plot above). As such, some of the purchases in this segment may also be commercial vehicles. As such, the 'ideal' car in the high-end market is a little fuzzier, though it tends not to be a automatic vehicle. That vehicle should have a good condition (though not necessarily 'excellent', though this could be a self-reporting bias) and have a lower number of miles.

#### How to generate better car sales

Choosing which market to operate in will dictate the average sale price and volume of sales to be expected. Both the low-end and high-end markets have a great deal of potential value, however cars which fit the profile of the high-end market tend to capture a larger population and a larger price. You can expect for higher-end vehicles to be sold <i>slightly</i> more consistently, and at a better price. In order to improve car sales, attempt to buy and sell vehicles which fit this profile.

Thanks,
Xavier

