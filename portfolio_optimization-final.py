import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# 1. VERİ HAZIRLIĞI (Aynı kalıyor)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
data = yf.download(tickers, period="1y", interval="1d", auto_adjust=True)

try:
    prices = data['Close']
except KeyError:
    prices = data['Adj Close']

returns = prices.pct_change().dropna()

# Yıllıklandırma faktörleri
mu = returns.mean() * 252
Sigma_sample = returns.cov() * 252

# 2. MODEL 1 & 2: KLASİK VE REGULARIZED (Zaten yapmıştık, hızlıca tekrar kuralım)
# Ağırlıkların toplamı 1 olacak şekilde normalize etme fonksiyonu
def get_analytical_weights(Sigma, mu):
    inv_Sigma = np.linalg.inv(Sigma)
    ones = np.ones(len(mu))
    w = inv_Sigma @ mu
    return w / (ones @ w)

# Klasik
w_naive = get_analytical_weights(Sigma_sample, mu)

# Regularized (Ledoit-Wolf)
lw = LedoitWolf()
Sigma_lw = lw.fit(returns).covariance_ * 252
w_regularized = get_analytical_weights(Sigma_lw, mu)

# 3. MODEL 3: CONSTRAINED OPTIMIZATION (Long-Only)
# Burası "Inverse Problem"den "Optimization Problem"e geçtiğimiz yer.
# Amaç: Sharpe Oranını Maksimize Et (Negatif Sharpe'ı Minimize et)
# Kısıtlar: Ağırlıklar toplamı 1, Her ağırlık 0 ile 1 arasında.

def negative_sharpe(w, mu, Sigma):
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w.T)
    return -ret / vol # Sharpe'ı maksimize etmek için negatifini minimize ediyoruz

# Kısıtlar ve Sınırlar
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) # Toplam = 1
bounds = tuple((0, 1) for _ in range(len(tickers))) # 0 <= w <= 1 (Short Yasak!)

# Başlangıç tahmini (Eşit dağılım)
init_guess = np.array([1/len(tickers)] * len(tickers))

# Çözücü (Solver) Çalıştır
opt_result = minimize(negative_sharpe, init_guess, args=(mu, Sigma_lw),
                      method='SLSQP', bounds=bounds, constraints=constraints)

w_constrained = opt_result.x

# 4. SONUÇ RAPORU
results = pd.DataFrame({
    'Hisse': tickers,
    'Klasik (Naive)': np.round(w_naive * 100, 1),
    'Robust (Reg.)': np.round(w_regularized * 100, 1),
    'Real-World (Long-Only)': np.round(w_constrained * 100, 1)
})

print("\n--- PROFESYONEL PORTFÖY SENARYOLARI ---")
print(results)

# 5. GÖRSELLEŞTİRME (3 Sütunlu)
x = np.arange(len(tickers))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, w_naive, width, label='Klasik (Math Only)', color='salmon', alpha=0.6)
rects2 = ax.bar(x, w_regularized, width, label='Robust (Regularized)', color='royalblue', alpha=0.8)
rects3 = ax.bar(x + width, w_constrained, width, label='Ticari (Long-Only)', color='forestgreen')

ax.set_ylabel('Portföy Ağırlığı')
ax.set_title('Portföy Optimizasyonu: Teoriden Pratiğe Evrim\n(Inverse Problem -> Regularization -> Constrained Optimization)')
ax.set_xticks(x)
ax.set_xticklabels(tickers)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=1)

# Değerleri çubukların üzerine yazalım
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

autolabel(rects3) # Sadece final sonucu etiketleyelim kalabalık olmasın

# Grafiği yüksek kalitede kaydet (README için)
plt.savefig('portfolio_scenarios.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()