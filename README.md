# Demand Forecasting & Procurement Optimization
### Newsvendor-based ordering strategy for *Le Club Français du Vin*
 
> Academic project — 
> **Authors:** Blengbi Kanga · Sophia Troquereau · Jasmine Petrosyan · Selma Zmiri
> **Stack:** Python · NumPy · SciPy · Matplotlib · Jupyter
 
---
 
## 1. Executive summary
 
Le Club Français du Vin sells wine through a catalog, ordering each wine months ahead of the sales season. Ordering too little means lost sales; ordering too much means bottles stored for months and sold at a discount. This project builds a data-driven ordering policy on top of the club's existing forecasts.
 
| Key finding | Value |
|---|---|
| Forecast bias (mean actual / forecast) | **0.865** → forecasts overshoot demand by ~13 % |
| Forecast dispersion (std of A/F ratio) | **0.393** (~45 % of the mean) → unreliable at the single-wine level |
| Wines that sold less than forecast | **27 of 40** in the historical data |
| Main driver of the optimal order size | **Retail price** (through the cost of underage/overage), more than forecast volume |
| Where the model breaks | Bottles priced above ~€26 (red) / ~€41 (white/rosé), where overage cost turns negative |
 
**Recommendation:** correct forecasts by the historical bias, model demand as a normal distribution, and size each order with a critical-ratio (newsvendor) rule. Then revisit the rule as soon as customers can be redirected to substitutes (online ordering).
 
---
 
## 2. Business context & objectives
 
- Wines are ordered **once, well in advance** (storage of 8 months for white/rosé, 15 months for red).
- Unsold bottles are **liquidated at a discount** (40 % for white/rosé, 30 % for red).
- Demand is uncertain and forecasts are known to be imperfect.
- 
**Objectives**
 
1. Quantify the quality and bias of the existing forecasts.
2. Turn point forecasts into a demand *distribution* for each wine.
3. Derive the cost of ordering too little (Cu) and too much (Co) for each wine.
4. Recommend order quantities balancing **expected profit** and **service level** (probability of being in stock).
5. Stress-test the framework (extreme prices, online substitution, a "perfect substitute" wine).
---
 
## 3. Methodology
 
### 3.1 Forecast accuracy
 
For each wine in the historical data, the ratio of actual demand to forecast is computed:
 
$$ r_w = \frac{\text{actual demand}_w}{\text{forecast}_w} $$
 
The mean $\bar r$ measures **bias**; the standard deviation $s_r$ measures **reliability**.
 
### 3.2 Demand model
 
For a wine with forecast $F_w$, demand is modeled as normally distributed:
 
$$ \mu_w = F_w \cdot \bar r \qquad \sigma_w = F_w \cdot s_r $$
 
### 3.3 Cost structure
 
| Parameter | White / rosé | Red |
|---|---|---|
| Storage duration $m$ (months) | 8 | 15 |
| Clearance discount $d$ | 40 % | 30 % |
| Storage cost | €0.10 / bottle / month | €0.10 / bottle / month |
| Shipping | €1.25 / bottle | €1.25 / bottle |
| Purchase cost | 50 % of retail price $p$ | 50 % of retail price $p$ |
 
**Cost of underage** (margin lost on each unmet unit of demand):
 
$$ C_u = 0.5\,p - 1.25 $$
 
**Cost of overage** (loss on each unsold bottle after clearance):
 
$$ C_o = p\left(0.5 - (1-d) + \tfrac{0.15\,m}{24}\right) + 0.1\,m + 1.25 $$
 
### 3.4 Order quantity
 
The critical ratio and the profit-maximizing (newsvendor) quantity are:
 
$$ CR_w = \frac{C_u}{C_u + C_o} \qquad Q^*_w = \mu_w + \sigma_w \cdot \Phi^{-1}(CR_w) $$
 
For a target service level $\alpha$ (here 95 %), the quantity is $Q_\alpha = \mu_w + \sigma_w \cdot \Phi^{-1}(\alpha)$.
 
```python
mean_demand[w] = forecast[w] * mean_ratio
std_demand[w]  = forecast[w] * std_ratio
cr = Cu[w] / (Cu[w] + Co[w])
Q_profit = mean_demand[w] + std_demand[w] * stats.norm.ppf(cr)
Q_95     = mean_demand[w] + std_demand[w] * stats.norm.ppf(0.95)
```
 
---
 
## 4. Results
 
### 4.1 The forecasts are biased and unstable
 
| Metric | Value |
|---|---|
| Wines in historical data | 40 |
| Mean A/F ratio | 0.8653 |
| Std of A/F ratio | 0.3928 |
| Range | 0.17 (Montagne St-Émilion) – 2.34 (Côtes du Vivarais) |
| Interval mean ± 1 std | [0.47 ; 1.26] |
 
- **Negative bias:** on average, actual demand is only ~87 % of the forecast. Ordering straight from raw forecasts would generate excess inventory, tied-up cash and waste (wine has a shelf life).
- **Low reliability:** with a coefficient of variation around 45 %, some wines sell far less than planned while others exceed the forecast by more than 25 %.
- **Practical fix:** scale new forecasts by **0.865**, while keeping a wide safety margin.
### 4.2 Resulting demand distributions
 
Applying the ratios to the 30 wines of the prediction set:
 
| Metric | Value |
|---|---|
| Average expected demand across wines | 2,684 bottles |
| Average std across wines | 1,218 bottles |
| Std / mean, for every wine | ≈ 0.45 (constant by construction) |
 
The catalog-level averages are of limited use: decisions must be made **wine by wine**. Because the same relative uncertainty is applied to every wine, the wine-level differences in optimal order size come from **volume and cost structure**, not from wine-specific volatility.
 
> **Note:** ordering exactly the mean demand gives a ~50 % stock-out probability under a normal model. Whether to order above or below the mean depends on the critical ratio.
 
### 4.3 Price drives the cost trade-off
 
<!-- ![Price vs Cu](figures/price_vs_cu.png) -->
<!-- ![Price vs Co](figures/price_vs_co.png) -->
 
**Price vs Cu — perfectly linear and increasing.** Since $C_u = 0.5p - 1.25$, all wines sit on one line. Missing a sale of the most expensive wine (Aloxe-Corton, €21.90, Cu = €9.70) costs about 25× more than missing one of the cheapest (VDP Ardèche, €3.25, Cu = €0.38).
 
**Price vs Co — decreasing.** Fixed costs (storage, shipping) weigh heavily on cheap bottles (Co ≈ €2.0–2.4) and become marginal on expensive ones (Co < €1 above ~€19).
 
**Red vs white.** Red wines carry a higher Co at low prices (longer storage: 15 vs 8 months). The gap reverses around **€12.4**: above that price, the smaller discount on reds (30 % vs 40 %) makes them *cheaper* to overstock than whites.
 
**Implication.** Since Cu rises and Co falls with price, the critical ratio increases steeply with price:
 
- cheap wines → conservative orders (often below mean demand),
- expensive wines → aggressive stocking.
Price is the primary driver of the ordering strategy, ahead of forecast volume.
 
### 4.4 Recommended order quantities
 
Six wines were analyzed (rosé priced with white-wine parameters). All quantities use the 95 % service-level target as the reference.
 
| Wine | Type | Price | Cu | Co | CR | Q* (max profit) | Q (95 % in stock) | **Selected Q** |
|---|---|---|---|---|---|---|---|---|
| Gigondas La Payouse | Red | €13.90 | 5.70 | 1.27 | 0.82 | 1,221 | 1,512 | **1,512** |
| Graves Ch. Haut Pommarède | Red | €8.40 | 2.95 | 1.86 | 0.61 | 979 | 1,512 | **1,512** |
| VDP Coteaux de l'Ardèche Réserve Rouge | Red | €3.25 | 0.38 | 2.40 | 0.13 | 1,511 | 5,290 | **1,512** |
| Cabernet d'Anjou Goulaine | Rosé | €5.60 | 1.55 | 1.77 | 0.47 | 2,498 | 4,534 | **2,498** |
| Pessac-Léognan Ch. Haut Nouchet | Red | €18.90 | 8.20 | 0.74 | 0.92 | 1,832 | 1,965 | **1,965** |
| Côtes de Bourg Ch. Florimond | Red | €7.20 | 2.35 | 1.99 | 0.54 | 1,179 | 1,965 | **1,965** |
 
*Prices are back-computed from Cu. Order quantities are rounded up.*
 
**Decision rule applied**
 
- **Cu > Co** → running out is the costlier mistake → order for the service level (`get_Quantity_for_InStock`, α = 95 %).
- **Cu < Co** → overstocking is the costlier mistake → order for maximum profit (`get_Quantity_for_Profit`).
**Optional fine-tuning.** When Cu > Co but the 95 % service level already makes stock-outs rare, the order can be moved slightly back toward the profit-maximizing quantity. For Gigondas: (1,512 − 1,221) × 5 % ≈ 15 bottles, giving **1,497**.
 
### 4.5 Same demand, different orders
 
Gigondas and Graves share the same demand distribution (μ ≈ 865, σ ≈ 393) but get different quantities. Only their cost structure differs: Gigondas has a higher price, hence a higher critical ratio (0.82 vs 0.61) and a higher profit-maximizing quantity (1,221 vs 979). **Same demand risk, different cost of being wrong → different optimal order.**
 
---
 
## 5. Stress-testing the model
 
### 5.1 When the theory breaks: a €60 bottle
 
| | Cu | Co | CR |
|---|---|---|---|
| Red, €60 | 28.75 | **−3.62** | 1.144 |
| White, €60 | 28.75 | **−0.95** | 1.034 |
 
A negative Co means an unsold bottle still returns more than it cost, even after the discount, storage and shipping. This has several consequences:
 
- CR exceeds 1, which is outside the domain of the inverse normal CDF: `get_Quantity_for_Profit` returns `nan`.
- The overstock/understock trade-off disappears, so the model has no finite optimum.
- Within the model, ordering more is always better, so the framework no longer applies.
Under the assumed parameters, Co turns negative above roughly **€26 for red** and **€41 for white/rosé**. In practice, capacity, cash constraints and limited demand for discounted bottles would still cap orders, which the model does not capture.
 
### 5.2 Online ordering with stock-out notifications
 
If customers are told a wine is sold out and can switch to another one, the "lost sale" assumption no longer holds:
 
- **Cu falls for every wine**, because part of the margin is recovered on the substitute. Lower Cu → lower CR → lower Q across the catalog.
- **Demands become interdependent.** A stock-out on wine A raises demand for wine B, so wine-by-wine optimization is no longer valid and the catalog must be optimized as a whole.
- The profit curve is flat near the optimum. For Pessac-Léognan, ordering for a 90 % service level (Q = 1,779) instead of the optimum (Q* = 1,832, 91.7 % in stock) costs about €4 of expected profit (€8,522 vs €8,526) while saving 53 bottles of inventory.
**Conclusion:** the club should order less than the single-wine optimum and accept lower individual service levels, since substitution protects the catalog-level service.
 
### 5.3 The "ideal substitute" wine (*Subs Parfait*)
 
If one wine absorbs most customers who cannot find their first choice, its demand has two components:
 
$$ D_{SP} = D_{\text{own}} + \sum_{w} \beta_w \,(D_w - Q_w)^+ $$
 
where $\beta_w$ is the share of stocked-out customers of wine $w$ who switch to Subs Parfait, and $(D_w - Q_w)^+$ is the shortfall on wine $w$.
 
- **Its order quantity** must be computed from this combined demand, using the expected stock-outs across the whole catalog, rather than from its own forecast alone.
- **Effect on other wines:** a reliable substitute lowers Cu everywhere, so every other order can be leaner.
- **Circularity:** Subs Parfait's demand depends on the other orders, and the other orders depend on Subs Parfait. This calls for a joint optimization (or an iterative solution) rather than sequential decisions.
---
 
## 6. Limitations & next steps
 
- **Normal-demand assumption:** the normal distribution can produce negative demand and ignores skewness. Consider a lognormal or empirical distribution.
- **Constant uncertainty:** the same relative std is applied to all wines. Estimating volatility by wine category or price band would be more realistic.
- **Heuristic service-level rule:** the newsvendor optimum $Q^*$ already balances Cu and Co. Switching to the 95 % quantity whenever Cu > Co can inflate orders a lot for a small asymmetry (e.g. Côtes de Bourg, CR = 0.54: +67 % vs Q*). Comparing expected profit at both quantities before deciding is preferable.
- **Small historical sample:** 40 wines to estimate both bias and dispersion.
- **Independence across wines:** no cross-wine correlation or substitution is modeled in the base case.
- **Extensions:** multi-wine optimization with substitution matrices, capacity/budget constraints, and simulation of expected profit under alternative policies.
---
 
## 7. Reproducing the analysis
 
The analysis was run in a Jupyter notebook (Python 3, `numpy`, `scipy.stats`, `matplotlib`). Main steps:
 
1. Compute A/F ratios from the historical data, then their mean and standard deviation.
2. Build `mean_demand` and `std_demand` for each wine of the prediction set.
3. Compute `Cu` and `Co` from price, color, storage duration and discount.
4. Compute `get_Quantity_for_Profit` and `get_Quantity_for_InStock` per wine.
> **Data note:** wine names containing accents may be mis-encoded when loaded (e.g. `CÔTES` read as `CÃ”TES`). Match on the encoded string or normalize the encoding on import.
 
