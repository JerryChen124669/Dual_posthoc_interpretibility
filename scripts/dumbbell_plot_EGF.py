import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

response_name = {3: 'pERK', 18: 'pEGFR', 33: 'pAKT', 48: 'pFAK',
                 63: 'FoxO3a', 78: 'pS6', 93: 'FoxO1', 108: 'pMTOR',
                 123: 'pRSK', 153: 'pGSK3B', 168: 'pMEK', 183: 'pPI3K'}

# colors = ['#add8e6', '#87ceeb', '#4682b4']
color_all = list(mcolors.TABLEAU_COLORS.values())
colors = [color_all[0], color_all[3], color_all[2]]

data = {'Category': ['pERK', 'pEGFR', 'pAKT', 'pFAK', 'FoxO3a', 'pS6',
                     'FoxO1', 'pMTOR', 'pRSK', 'pGSK3B', 'pMEK', 'pPI3K'] * 3,
        'Linear': [0.6856, 0.9222, 0.6398, 0.7455, 0.8613, 0.8353,
                   0.7591, 0.8868, 0.7840, 0.6866, 0.9117, 0.7861,
                   0.8905, 0.9210, 0.8486, 0.6550, 0.8556, 0.8420,
                   0.7416, 0.8876, 0.9143, 0.8798, 0.9156, 0.7665,
                   0.8987, 0.8298, 0.6909, 0.7872, 0.8849, 0.7386,
                   0.7898, 0.9188, 0.8679, 0.9395, 0.8385, 0.8558],
        'Non-linear': [0.7632, 0.9355, 0.7068, 0.7580, 0.8661, 0.8823,
                       0.7874, 0.9073, 0.9073, 0.7729, 0.9416, 0.8229,
                       0.9284, 0.9362, 0.8637, 0.6725, 0.8628, 0.8978,
                       0.7735, 0.9087, 0.9426, 0.9044, 0.9269, 0.8117,
                       0.9107, 0.8483, 0.7200, 0.7953, 0.8875, 0.8453,
                       0.8217, 0.9210, 0.8942, 0.9451, 0.8455, 0.8713],
        'Color': [colors[0]] * 12 + [colors[1]] * 12 + [colors[2]] * 12
        }

df = pd.DataFrame(data)
df['Difference'] = df['Non-linear'] - df['Linear']
df = df.sort_values(by='Difference')

plt.rcParams['xtick.labelsize'] = 14
fig, ax = plt.subplots(figsize=(10, 20))

for i in range(df.shape[0]):
    ax.axhspan(i - 0.45, i + 0.45, color=df['Color'].iloc[i], alpha=0.2)

for i in range(df.shape[0]):
    ax.plot([df['Linear'].iloc[i], df['Non-linear'].iloc[i]], [i, i], color='gray', lw=1, zorder=1)

scatter1 = ax.scatter(df['Linear'], range(df.shape[0]), color='blue', label='Linear', zorder=2)
scatter2 = ax.scatter(df['Non-linear'], range(df.shape[0]), color='red', label='Non-linear', zorder=3)

ax.set_yticks(range(df.shape[0]))
ax.set_yticklabels(df['Category'], fontsize=16)
# ax.set_xticklabels(ax.get_xticks(), fontsize=16)
# plt.xticks(rotation=45)
ax.set_xlabel('Pearson Correlations', fontsize=16)
ax.set_title('Different Responses with different EGF concentrations', fontsize=18)

legend_handles = [mpatches.Patch(color=colors[0], label='EGF=1ng/ml', alpha=0.2),
                  mpatches.Patch(color=colors[1], label='EGF=10ng/ml', alpha=0.2),
                  mpatches.Patch(color=colors[2], label='EGF=100ng/ml', alpha=0.2)]
legend1 = ax.legend(handles=legend_handles, loc='lower left', fontsize=16)
legend2 = ax.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=16)
ax.add_artist(legend1)
ax.add_artist(legend2)
# ax.legend(handles=legend_handles, fontsize=16)

plt.tight_layout()
# plt.show()
plt.savefig(f'plot/dumbbell.svg', format='svg')
