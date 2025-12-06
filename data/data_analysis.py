import pandas as pd
import matplotlib.pyplot as plt


def prep_csv(src, dst):
    df = pd.read_csv(src)

    columns = [
        'NACCID',
        'NACCAGE',
        'SEX',
        'RACE',
        'EDUC',
        'NACCFAM',
        'OTHMUT',
        'OTHMUTX',
        'NACCADMU',
        'NACCGDS',
        'CDRSUM',
        'APA',
        'AGIT',
        'HYPERTEN',
        'DIABETES',
        'STROKE',
        'CBTIA',
        'CVD',
        'CVDIF',
        'TOBAC100',
        'SMOKYRS',
        'TOBAC30',
        'PACKSPER',
        'NACCBMI',
        'NACCMMSE',
        'NACCMOCA',
        # 'REY1REC',
        # 'REY2REC',
        # 'REY3REC',
        # 'REY4REC',
        # 'REY5REC',
        # 'REY6REC',
        # 'REY1INT',
        # 'REY2INT',
        # 'REY3INT',
        # 'REY4INT',
        # 'REY5INT',
        # 'REY6INT',
        # 'REYDREC',
        # 'REYDINT',
        # 'REYTCOR',
        # 'REYFPOS',
        'EVENT',
        'TIME_TO_EVENT'
    ]

    df = df[columns]
    to_drop = (df['EVENT'] == 1) & (df['TIME_TO_EVENT'] == 0)
    print(f'Truncating df - Dropping {len(to_drop)} rows')
    df = df[~to_drop]

    df.to_csv(dst, index=False)

    return df




# df = prep_csv(src='data/ad_lastvisit_with_time.csv', dst='data/NACC_proc.csv')
df = pd.read_csv('data/NACC_proc.csv')


# Statistics
n_censored = len(df) - sum(df['EVENT'])
prop_censored = n_censored / len(df)
print('Prop Censored:', prop_censored)

times = df['TIME_TO_EVENT']
plt.figure(dpi=300)
plt.hist(times, bins=25)
plt.title('')
plt.show()