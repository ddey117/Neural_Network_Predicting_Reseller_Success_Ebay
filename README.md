 # Predicting Resale Value of Knives from a Texas Government Surplus Store
## Using Machine Learning to Support an Ebay Store's Financial Success



## Project Links

Below is the link for the GitHub project page. 

[Github link](https://github.com/ddey117/Neural_Network_Predicting_Reseller_Success_Ebay)



**Author:** Dylan Dey


# Overview
[Texas State Surplus Store](https://www.tfc.texas.gov/divisions/supportserv/prog/statesurplus/)

[What happens to all those items that get confiscated by the TSA? Some end up in a Texas store.](https://www.wfaa.com/article/news/local/what-happens-to-all-those-items-that-get-confiscated-by-the-tsa-some-end-up-in-a-texas-store/287-ba80dac3-d91a-4b28-952a-0aaf4f69ff95)

[Texas Surplus Store PDF](https://www.tfc.texas.gov/divisions/supportserv/prog/statesurplus/State%20Surplus%20Brochure-one%20bar_rev%201-10-2022.pdf)

![Texas State Surplus Store](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYkwyu20VBuQ52PrXdVRaGRIIg9OPXJg86lA&usqp=CAU)

![Texas Knives In Stores](https://arc-anglerfish-arc2-prod-bostonglobe.s3.amazonaws.com/public/MWJCCFBSR4I6FCSNKONTFJIRAI.jpg)

[Everything that doesn't make it through Texas airports can be found at one Austin store](https://cbsaustin.com/news/local/everything-that-doesnt-make-it-through-texas-airports-can-be-found-at-one-austin-store)


> The Texas Facilities Commission collects left behind possessions, salvage, and surplus from Texas state agencies such as DPS, TXDOT, TCEQ, and Texas Parks & Wildlife. Examples of commonly available items include vehicles, furniture, office equipment and supplies, small electronics, and heavy equipment. The goal of this project is to create a predictive model in order to determine the resale value of knivse from the Texas State Surplus Store on eBay to support the financial health of an eBay storefront. 


# Business Problem

[Family Ebay Store Front](https://www.ebay.com/str/texasdave3?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=ZW3G27tGR_m&sssrc=3418065&ssuid=&widget_ver=artemis&media=COPY)

![Father's Ebay Account Since 1999](texas_dave.jpg)

[Texas Dave's Knives](https://www.ebay.com/str/texasdave3/Knives/_i.html?store_cat=3393246519)

3 Major Painpoints:
<ol>
  <li>Selection of an item to sell: 
      <ul>
          <li>The item should be readily available in decent condition for the seller to purchase at a low price but not so widely available that the market is saturated with that item.</li>
          <li>there needs to be a demand for the item.</li>
          <li>Be useful to a large number of people even in a used condition.</li>
          <li>Be easy, cost effective and safe to ship.</li>
      </ul>
    </li>
  <li>Buying at a cost low enough to make a profit: 
    <ul>
          <li>Ebay takes about 13.5% of the sold price for brokering the sale.</li>
          <li>Shipping costs and overhead fees can add up.</li>
      </ul>
    </li>
  <li>The cost of excess inventory: 
    <ul>
          <li>A seller can obtain quality items at a reasonable cost and then the inventory may sit with no sales, meaning the capital expended is sitting tied up in unwanted items. This inventory carry cost is a drain on profitability.</li>
      </ul>
    </li>
</ol>


> A potential target that has been selling decently on the eBay store front from random sampling is something that has been overflowing out of bins at the State Surplus Store for years: pocketknives. Does this target answer all of the pain points above?


> I have been experimenting with low cost used knives for resale but have not risked a large capital investment in the higher end items. The goal of this project is to attempt to address the pain points to determine if a larger investment would pay off. Can I identify which knives are worth investing in so that I can turn a decent profit and hopefully avoid excess inventory? A data driven approach would help avoid costly mistakes from the "system"  currently employed, which seems to be mainly a gambler’s approach. By managing resources upfront through a model, I can effectively increase my return on investment with messy data such as pictures and titles.


> There are <mark>eight buckets</mark> of presorted brand knives that I was interested in, specifically. These bins are behind glass, presorted, branded(and therefore have specific characteristics and logos for my model to identify), and priced higher. However, the staff has a very large amount of confiscated items flowing into the facility to list for resale, and when that happens they will not have time to presort them and they end up in huge buckets of unsorted knives for people to dig through. The brands will be priced the same, they are just no longer sorted and harder to find. Expanding the bins to pull inventory from will increase the chance of finding inventory worth reselling, at the cost of time spent sorting the knives.

**sorted bucket example**
![case price image](images/casePriceBucket2.jpg)

**overflow example**
![overflow image](images/overflow.jpeg)

### Domain Understading: Cost Breakdown
- padded envelopes: \$0.50 per knife
- flatrate shipping: \$4.45 per knife
- brand knife at surplus store: 15, 20, 30, or 45 dollars per knife
- overhead expenses (gas, cleaning suplies, sharpening supplies, etc): $3.00
- Ebay's comission, with 13\% being a reasonable approximation

| Brand       | Cost at Surplus Store | Cost Adjusted for Expenses |
| :---        |        :----:         |                       ---: |
| Benchmade   | $45.00                | $52.95 + 13% comission     |
| Buck        | $20.00                | $27.95 + 13% comission     |
| Case        | $20.00                | $27.95 + 13% comission     |
| CRKT        | $15.00                | $22.95 + 13% comission     |
| Kershaw     | $15.00                | $22.95 + 13% comission     |
| SOG         | $15.00                | $22.95 + 13% comission     |
| Spyderco    | $30.00                | $37.95 + 13% comission     |
| Victorinox  | $20.00                | $27.95 + 13% comission     |

[Ebay Developer Website](https://developer.ebay.com/)
> Ebay has a website for developers to create an account and register an application keyset in order to make API call requests to their live website. By making a findItemsAdvanced call to the eBay Finding APIVersion 1.13.0, I was able to get a large dataset of [category_id=<48818>](https://www.ebay.com/sch/48818/i.html?_from=R40&_nkw=knife) knives listed for sale. This data is limited to anything listed within the past 90 days from when the API call was made.


> The eBay Finding APIVersion 1.13.0 [findItemsAdvanced](https://developer.ebay.com/devzone/finding/callref/finditemsadvanced.html) and The eBay Shopping APIVersion 1247 [GetMultipleItems](https://developer.ebay.com/Devzone/shopping/docs/CallRef/GetMultipleItems.html) were used to collect data from listed eBay posts.


> All of the data gathered from eBay's public API is limited to listed data posted in the past 90 days and doesn't include a "sold" price. Sold data is only available to seller accounts on a webpapp known as Terapeak. The sold data goes back 2 years.

>A majority of the data was scraped from Terapeak, as this data goes back 2 years as compared to the API listed data that only goes back 90 days. It is assumed a large enough amount of listed data should approximate sold data well enough to prove useful for this project. 


<div class="alert alert-block alert-info">  
<span style="color:green">After cleaning the data to remove lots, groups, and most of the listings with multiple knives using regex, there were over 70,000 listings scraped from Terapeak, represting the final sale price of 70K different used knives and their associated titles and images.</span>   
 <span style="color:yellow">Around 10K listings were obtained using eBay's API, represnting current listed used knives and their asking price. A large enough sample size of knives currently listed for sale should approximate the true value of the knife. A lot more of the sold data was available and is a clear indicator of the value of knives over the past two years.</span>
 </div>
 
<div class="alert alert-block alert-warning">  
After combining the two datasets mentioned above, the data was then filtered to remove outliers using IQR filtering. The final dataset includes roughly 76K listings for knives between the 8 brands of interest.
 </div>
 
#### feature creation
<ul>
    <li>converted price: Combined sold price with shipping in USD.</li>
    <li>profit: Combined sold price with shipping in USD minus the cost of the knife at the Surplus Store,Ebay's comission, flat shipping cost for a 0.5 lb item and cost of envelope, and overhead costs.</li>
     <li>Return On Investment(ROI): calculated by dividing the profit earned on an investment by the cost of that investment, expressed in %USD.</li>
</ul> 

## Methods

>This project utilizes descriptive analysis to compare the profitability and return on investment for the 8 different brands avaialable for sale at the Texas State Surplus store. By examining different measures of centrality for previous prices sold on eBay in the past 2 years, the measure of variability, percentiles and also the construction of tables & graphs, it can be inferred which brands of knives to invest more heavily into and which ones to ignore completely at just a glance.


> On top of descriptive analysis, the data will be used to train two different Neural Network models to predict the resale value of knives. The target feature for the model to predict is the total price (shipping included) that a knife should be listed on eBay. One model will use an RNN on titles in order to find potential listings that are undervalued and could be worth investing in. Another model will accept only images as input, as this is an input that can easily be obtained in person at the store. This model will use past sold data of knives on eBay in order to determine within an acceptable amount of error the price it will resell for on eBay (shipping included) using only an image. 

# Results


---------------------------------------------------
### Recurrent Neural Network (GRU)

The GRU Price Predictive Model is recommended for use when listing a pocket knife on sale to help list it appropriately.

**Best Mean Absolute Error: $14.28**

---------------------------------------------------

### Convoluted Neural Network on Grayscale Images

- The MAE when testing the CNN was roughly \\$25.00. That is an error of plus or minus about 50\% of the mean price of knives sold. Not acceptable yet as compared to the RNN with titles. Will address in future work.

![GRU regPlot](https://github.com/ddey117/Neural_Network_Predicting_Reseller_Success_Ebay/blob/master/images/RNN/regPlot_GRU_performance.png?raw=true)

# Business Recommendations
Summary: Invest in Case brand and Spyderco knives at the Texas Surplus store. Benchmade knives will return high profits as well but lower return on investment. Deploy the GRU predictive modeling network when listing a pocket knife for sale on eBay to help balance time, excess inventory costs, and lost revenue. 

## Descriptive Analysis Conclusion

**Risk Tolerance/Available Capital**

Depending on how much upfront capital invested and if you can tolerate addition risk, Benchmade knives have returned the highest profit in the past 2 years when compared to the other brands avalaible at the Surplus Store. 

**Volume of Sales**

It is important to consider the volume of knives sold in order to avoid excess inventory. Benchmade knives have high profit but are rarely listed/sold compared to other brands. They are more rare than the other knives and past data suggest they will likely be harder to find on average when compared to the other brands.

**Risk/Initial Cost continued**

On the other hand, Case knives have demonstrated the lowest risk and highest percentage for return on investment in the past 2 years on eBay and has had the highest daily volume of sales. 


**Balancing Act**

Spyderco knives offer a great compromise between ROI and pure profit. Investing in this brand should return high profits without risking as much capital. There is also a reasonable amount of daily volume for Spyderco brand knives sold. Case knives have almost double the daily volume of sales compared to Spyderco, and they will likely be easier to find. Case knives are also worth investing in to maximize ROI and efficiency searching at the Surplus Store for more inventory to move.

-------------------------------------

## Model Analysis Conclusion
 - The performance metrics for the GRU Price Predictive Model was the Mean Squared Error and Mean Absolute Error for the model when predicting for the correct price to list the knife (an approximation for the true value of the knife). The best performing Model exhibited a Mean Absolute Error of \\$14.28. I believe an error range of plus or minus \\$14.28 outperforms the current process of scrolling through webpages and having the lister try to guess themselves. Even given missing the correct value by about 14 dollars and a quarter for each knife listed on average, the time saved trying to figure out a correct price within an acceptable limit without the model has implicit cost that is hard to value on a spreadsheet.
 
 - Deploying the model helps balance excess inventory costs for too high of prices vs loss revenue.
    
**Summary: I reccomend deploying the Price Predicting Model before posting a pocket knife for sale on eBay.**
 
## Future Work
- Expand data to include other products readily purchasable at the Surplus Store. 

- Explode nested list of images as form of data augmentation to increase image input size by about 5-fold. Attempt other data augmentation techniques.

- Obtain more aspect related data for knives to understand most important cost altering features more thoroughly. Enough of this data could use a predictive model that isn’t a black box.


# THANK YOU



Dylan Dey

email: ddey2985@gmail.com



```
├── README.md                     <- The top-level README for reviewers of this project
├── Data_Sourcing_Notebook.ipynb  <- Notebook for obtaining and processing Ebay data
├── Data_Exploration.ipynb        <- Descriptive Analysis of data
├── Model_Interpret.ipynb         <- modeling notebook
├── Project_Presentation.pdf      <- project presentation pdf
├── listed_data                   <- Both sourced externally and generated from code
├── sold_data                     <- Both sourced externally and generated from code
└── images                        <- Graphs and visualization/various helpful images

