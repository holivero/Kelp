In this repository we share the implementation of the numerical scheme proposed in the article "Age-structured stochastic populations under dynamic  harvesters' behavior: well-posedness, asymptotic stability and numerically-amenable approximations" by M. Isidora Ávila-Thieme, Kerlyns Martínez, Héctor Olivero, Mauricio Tejo, and Leonardo Videla.

# Some animations
Using the script compare_scenarios.py we compare the evolution of the model for different values for the Subsidy parameter and the volatility of the price.

## Subsidy
We used three levels: 0 (clp), 150 (clp) and 300 (clp). Bellow we show the output of the simulation.

### Juveniles vs. Adults under harvesting with dynamic compliance
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySubsidy/0/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/150/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/300/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Subsidy = 0</p>
</td>
<td align="center">
<p>Subsidy = 150</p>
</td>
<td align="center">
<p>Subsidy = 300</p>
</td>
</tr>
</table>
</div>


---


### Juveniles vs. Adults under harvesting with full compliance
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySubsidy/0/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/150/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/300/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Subsidy = 0</p>
</td>
<td align="center">
<p>Subsidy = 150</p>
</td>
<td align="center">
<p>Subsidy = 300</p>
</td>
</tr>
</table>
</div>

---

### Juveniles vs. Adults under harvesting with dynamic compliance, full compliance and non harvesting
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySubsidy/0/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/150/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/300/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Subsidy = 0</p>
</td>
<td align="center">
<p>Subsidy = 150</p>
</td>
<td align="center">
<p>Subsidy = 300</p>
</td>
</tr>
</table>
</div>

---

### Complince (E) vs Total population (Adults + Juveniles)
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySubsidy/0/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/150/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySubsidy/300/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Subsidy = 0</p>
</td>
<td align="center">
<p>Subsidy = 150</p>
</td>
<td align="center">
<p>Subsidy = 300</p>
</td>
</tr>
</table>
</div>

## Price volatility 
We used three levels: 0.07, 0.09 and 0.11. Bellow we show the output of the simulation.

### Juveniles vs. Adults under harvesting with dynamic compliance
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySigmaPrice/007/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/009/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/011/LevelSetHistogram_J_vs_A.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Price volatility  = 0.07</p>
</td>
<td align="center">
<p>Price volatility  = 0.09</p>
</td>
<td align="center">
<p>Price volatility  = 0.11</p>
</td>
</tr>
</table>
</div>


---


### Juveniles vs. Adults under harvesting with full compliance
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySigmaPrice/007/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/009/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/011/LevelSetHistogram_J_vs_A_fc.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Price volatility  = 0.07</p>
</td>
<td align="center">
<p>Price volatility  = 0.09</p>
</td>
<td align="center">
<p>Price volatility  = 0.11</p>
</td>
</tr>
</table>
</div>

---

### Juveniles vs. Adults under harvesting with dynamic compliance, full compliance and non harvesting
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySigmaPrice/007/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/009/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/011/LevelSetHistogram_J_vs_A_scenarios.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Price volatility = 0.07</p>
</td>
<td align="center">
<p>Price volatility  = 0.09</p>
</td>
<td align="center">
<p>Price volatility  = 0.11</p>
</td>
</tr>
</table>
</div>

---

### Complince (E) vs Total population (Adults + Juveniles)
<div align="center">
<table>
<tr>
<td>
<img src="animations/comparingBySigmaPrice/007/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/009/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
<td>
<img src="animations/comparingBySigmaPrice/011/LevelSetHistogram_E_vs_TP.gif" width="100%"/>
</td>
</tr>
<tr>
<td align="center">
<p>Price volatility  = 0.07</p>
</td>
<td align="center">
<p>Price volatility  = 0.09</p>
</td>
<td align="center">
<p>Price volatility  = 0.11</p>
</td>
</tr>
</table>
</div>
