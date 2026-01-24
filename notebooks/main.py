from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from utils import *

def clean_nfl_string(text, sep='-', keep_left=True):
    if not isinstance(text, str):
        return None
    parts = text.split(sep)
    if len(parts) < 2:
        return text
    result = parts[0] if keep_left else parts[1]
    return result.strip()

def cleaned_data():
    p = Path(__file__).resolve()
    base = None
    for parent in p.parents:
        if (parent / "data").exists():
            base = parent
            break
    if base is None:
        base = p.parents[3]
    raw = base / "data" / "raw"

    try:
      
        df_passing = pd.read_csv(raw / "passing_data.csv")
        df_adv_passing = pd.read_csv(raw / "adv_passing.csv")
        df_trailing = pd.read_csv(raw / "qb_trailing_data.csv")

    except FileNotFoundError as e:
        print("Data file not found:", e)
        return
    df_trailing = df_trailing.rename(columns = {'Player Name': 'Player'})
    df_trailing['Player'] = df_trailing['Player'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df_trailing = df_trailing[['Player', 'EPA/Play', 'Success %']]
    df_trailing['Success %'] = df_trailing['Success %'].str.replace(r'%', '', regex=False)
    df_trailing['Success %'] = pd.to_numeric(df_trailing['Success %'], errors='coerce').fillna(0).astype(float)

    df_passing = df_passing[df_passing['Pos'] == 'QB']
    df_adv_passing= promote_first_row_to_header(df_adv_passing)
    df_passing = df_passing[df_passing['Team'] != '2TM']
    df_adv_passing = df_adv_passing[df_adv_passing['Team'] != '2TM']
    df_passing = df_passing.merge(df_adv_passing[['Player', 'Team', 'IAY/PA' , 'CAY/PA']], on=['Player', 'Team'], how='left')
    df_passing['QBrec'] = df_passing['QBrec'].apply(clean_nfl_string)
    df_passing['QBrec'] = pd.to_numeric(df_passing['QBrec'], errors='coerce')
    cols = ['Player', 'Team', 'QBrec', 'Att', 'GS', 'CAY/PA', 'IAY/PA', 'EPA/Play', 'Success %']
    df_passing['Player'] = (df_passing['Player']
                    .str.strip()
                    .str.replace(r"\b(?:Jr|Sr|[IVX]+)\.?\b|[.']", '', regex=True, flags=re.IGNORECASE)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip())
    df_passing['Att'] = df_passing['Att'].str.replace(r'*', '', regex=False)
    df_passing['Att'] = pd.to_numeric(df_passing['Att'], errors='coerce').fillna(0).astype(int)
    df_passing['CAY/PA'] = df_passing['CAY/PA'].str.replace(r'*', '', regex=False)
    df_passing['CAY/PA'] = pd.to_numeric(df_passing['CAY/PA'], errors='coerce').fillna(0).astype(float)
    df_passing['IAY/PA'] = df_passing['IAY/PA'].str.replace(r'*', '', regex=False)
    df_passing['IAY/PA'] = pd.to_numeric(df_passing['IAY/PA'], errors='coerce').fillna(0).astype(float)
    df_passing = df_passing.merge(df_trailing, on='Player', how='left')

 
    df_passing = df_passing[cols]
    df_passing = df_passing.sort_values(by=['Att'], ascending=False)
    team_map = {
        'NWE': 'NE',
        'KAN': 'KC',
        'SFO': 'SF',
        'TAM': 'TB',
        'GNB': 'GB',
        'NOR': 'NO',
        'LVR': 'LV',

    }
    df_passing['Team'] = df_passing['Team'].replace(team_map)
    df_passing = df_passing[df_passing['Att'] >= 200]
    df_passing = df_passing.dropna(subset=['EPA/Play'])
    df_passing = df_passing.reset_index(drop=True)
    print(df_passing.head(50))
    return df_passing

def create_qb_scatter(df):
    x = df['Success %']
    y = df['EPA/Play']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot small dots as reference points
    ax.scatter(x, y, alpha=0.3, s=50, c='lightgray', zorder=1)
    
    for idx, row in df.iterrows():
        ax.annotate(row['Player'], (row['Success %'], row['EPA/Play']), 
                   fontsize=8, alpha=0.8, xytext=(-22, 15), 
                   textcoords='offset points', zorder=2)
    
    logo_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "nfl logos"
    for idx, row in df.iterrows():
        logo_file = logo_path / f"{row['Team']}.png"
        if logo_file.exists():
            img = Image.open(logo_file)
            imagebox = OffsetImage(img, zoom=0.01)
            ab = AnnotationBbox(imagebox, (row['Success %'], row['EPA/Play']), 
                               frameon=False, zorder=3)
            ax.add_artist(ab)
    
    ax.set_xlabel('Success %')
    ax.set_ylabel('EPA/Play')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = cleaned_data()
    create_qb_scatter(df)