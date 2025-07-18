# 📘 Advanced Mathematical Models for Finance

This repository contains a collection of labs and a final project completed for the course **Advanced Mathematical Models for Finance**. The materials cover theoretical and practical implementations of quantitative models used in modern financial analysis and derivatives pricing.

## 📂 Contents

- `LAB(X)/` — A set of Jupyter notebooks, dataset and report from weekly lab sessions.
- `FINALPROJECT/` — A comprehensive project integrating techniques learned from the course.


## 🧪 Lab Summaries

### **Lab 1 — Efficient Frontier**
In this lab, we computed and visualized the efficient frontier for both two and four risky assets. We also identified the minimum variance portfolio using analytical and numerical methods, and applied the model to real stock data from Google, Amazon, Tesla, and Apple.

### **Lab 2 — Portfolio Optimization and CAPM**
This lab explores analytical and numerical solutions to mean-variance portfolio optimization under constraints, including the impact of investor preferences and the presence of a risk-free asset. It also introduces the Capital Market Line and CAPM, with empirical validation of beta coefficients and Jensen’s alpha using market data and linear regression.

###**Lab 3 — Bootstrapping Discount and Zero-Coupon Curves**
In this lab, we implemented a multi-step bootstrapping procedure to construct a term structure of interest rates using market instruments such as deposits, futures, and swaps. Discount factors and zero-coupon rates were computed and visualized, with sensitivity to various day count conventions.

###**Lab 4 — Option Pricing: Monte Carlo and Binomial Tree Methods**
This lab explore numerical pricing of European call options under the Black-Scholes/Black76 framework using two approaches: Monte Carlo simulation and the Binomial Tree model. The implementation covers direct sampling, Euler discretization, and recombining trees. Convergence behavior is analyzed and compared to the analytical Black-Scholes solution to validate both methods.


## 📈 Final Project Overview
###**Pricing and Hedging of a Structured Bond with Embedded Asian Option**
This project analyzes a structured bond issued by a bank, featuring a coupon linked to the performance of an Asian call option on ENEL stock. We model the hedging swap between the issuing bank and an investment bank, pricing all relevant cash flows using interest rate curves bootstrapped from market data. The project includes the valuation of the Asian component via Monte Carlo simulation with antithetic variates, and numerical sensitivity analysis for Delta, Vega, and DV01. A multi-instrument hedging strategy is developed to neutralize all three risks using ENEL stock, European call options, and interest rate swaps.

