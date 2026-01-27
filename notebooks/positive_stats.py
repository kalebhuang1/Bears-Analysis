from pathlib import Path
import re
from turtle import left
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from PIL import Image
from utils import *


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
    # --- 1. Define X and Y ---
    x = df['Success %']
    y = df['EPA/Play']
    
    # --- 2. Setup Averages and Distances ---
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Calculate distance from center to identify outliers
    # (Using the x and y variables we just defined)
    df['dist'] = np.sqrt((x - x_mean)**2 + (y - y_mean)**2)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background reference points
    ax.scatter(x, y, alpha=0, s=30, c='gray', zorder=1)
    
    logo_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "nfl logos"
    
    for idx, row in df.iterrows():
        # Using the specific row values for the current player
        current_x = row['Success %']
        current_y = row['EPA/Play']
        
        logo_file = logo_path / f"{row['Team']}.png"
        
        # Outliers sit on top of the pile (higher zorder)
        z_val = 10 + int(row['dist'] * 100)
        
        if logo_file.exists():
            img = Image.open(logo_file).convert("RGBA")
            imagebox = OffsetImage(img, zoom=0.0085) 
            ab = AnnotationBbox(imagebox, (current_x, current_y), 
                               frameon=False, zorder=z_val)
            ax.add_artist(ab)
            
        # Selective Annotation (Labeling outliers only)
        if row['dist'] > 0.04: 
            ax.annotate(row['Player'], (current_x, current_y), 
                       fontsize=6, fontfamily='Times New Roman', fontweight='semibold', alpha=0.9,
                       xytext=(0, 10), textcoords='offset points', 
                       ha='center', zorder=z_val + 1)
    
    # --- 3. Final Styling ---
    ax.set_xlabel('Success % (Offensive Efficiency)', fontfamily='Times New Roman')
    ax.set_ylabel('EPA/Play (Individual Efficiency)', fontfamily='Times New Roman')
    ax.set_title('NFL QBs: Efficiency When Trailing (Min 200 Att)', fontsize=14, pad=20, fontfamily='Times New Roman')
    
    # Vertical and Horizontal Mean Lines
    ax.axhline(y=y_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=x_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()

# 2. Add 'Elite' Region (Top Right)
# Anchor point is (x_mean, y_mean), then we calculate width/height to the edge
    elite_rect = patches.Rectangle((x_mean, y_mean), right - x_mean, top - y_mean, 
                               linewidth=0, facecolor='green', alpha=0.08, zorder=0)
    ax.add_patch(elite_rect)

# 3. Add 'Struggling' Region (Bottom Left)
# Anchor point is (left, bottom), width/height goes up to the means
    struggle_rect = patches.Rectangle((left, bottom), x_mean - left, y_mean - bottom, 
                                  linewidth=0, facecolor='red', alpha=0.08, zorder=0)
    ax.add_patch(struggle_rect)

    plt.tight_layout()
    plt.savefig("caleb_pro_stats.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = cleaned_data()
    create_qb_scatter(df)