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


def cleaned_data_blitz():
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
        df_blitz_data = pd.read_csv(raw / "qb_blitz_data.csv")

    except FileNotFoundError as e:
        print("Data file not found:", e)
        return
    
    df_passing = df_passing[df_passing['Pos'] == 'QB']
    df_passing = df_passing[df_passing['Team'] != '2TM']
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
    df_passing['Att'] = df_passing['Att'].str.replace(r'*', '', regex=False)
    df_passing['Att'] = pd.to_numeric(df_passing['Att'], errors='coerce').fillna(0).astype(int)



    df_blitz_data = df_blitz_data.rename(columns = {'Player Name': 'Player'})
    df_blitz_data['Player'] = df_blitz_data['Player'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df_blitz_data = df_blitz_data[['Player', 'Sack %', 'ADoT']]
    
    df_final_blitz = df_blitz_data.merge(df_passing[['Player', 'Team', 'Att']], on='Player', how='left')
    df_final_blitz['Sack %'] = df_final_blitz['Sack %'].str.replace(r'%', '', regex=False)
    df_final_blitz['Sack %'] = pd.to_numeric(df_final_blitz['Sack %'], errors='coerce').astype(float)
    df_final_blitz['ADoT'] = pd.to_numeric(df_final_blitz['ADoT'], errors='coerce').astype(float)
    df_final_blitz = df_final_blitz[df_final_blitz['Att'] >= 200]
    print(df_final_blitz)
    
    return df_final_blitz

def create_scatter_blitz(df):
    x = df['ADoT']
    y = df['Sack %']
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    df['dist'] = np.sqrt((x - x_mean)**2 + (y - y_mean)**2)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(x, y, alpha=0, s=30, c='gray', zorder=1)
    
    logo_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "nfl logos"
    
    for idx, row in df.iterrows():
        current_x = row['ADoT']
        current_y = row['Sack %']
        
        logo_file = logo_path / f"{row['Team']}.png"
        
        z_val = 10 + int(row['dist'] * 100)
        
        if logo_file.exists():
            img = Image.open(logo_file).convert("RGBA")
            imagebox = OffsetImage(img, zoom=0.009) 
            ab = AnnotationBbox(imagebox, (current_x, current_y), 
                               frameon=False, zorder=z_val)
            ax.add_artist(ab)
            
        if row['dist'] > 0.04: 
            ax.annotate(row['Player'], (current_x, current_y), 
                       fontsize=6.5, fontfamily='Times New Roman', fontweight='semibold', alpha=0.9,
                       xytext=(0, 10), textcoords='offset points', 
                       ha='center', zorder=z_val + 1)
    
    ax.set_xlabel('ADoT (Average Depth of Target)', fontsize = 16, fontfamily='Times New Roman')
    ax.set_ylabel('Sack %', fontsize = 16, fontfamily='Times New Roman')
    ax.set_title('NFL QBs: When Blitzed (Min 200 Att)', fontsize=20, fontweight = 'semibold', alpha = 0.7, pad=20, fontfamily='Times New Roman')
    
    ax.axhline(y=y_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=x_mean, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    ax.text(
    x=0+0.003,              
    y=y_mean+0.12,            
    s=f'Avg Sack %: {y_mean:.2f}', 
    color='black',
    va='center',         
    ha='left',           
    fontsize=9,
    alpha=0.9,
    transform=ax.get_yaxis_transform() 
)

    ax.text(
    x=x_mean + 0.26,            
    y=0+0.003,             
    s=f'Avg ADoT: {x_mean:.2f}', 
    color='black',
    va='bottom',         
    ha='center',         
    fontsize=9,
    alpha=0.9,
    transform=ax.get_xaxis_transform()
)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()

    bad_rect = patches.Rectangle((left, y_mean), x_mean - left, top - y_mean, 
                               linewidth=0, facecolor='red', alpha=0.08, zorder=0)
    ax.add_patch(bad_rect)

    good_rect = patches.Rectangle((x_mean, bottom), right - x_mean, y_mean - bottom, 
                               linewidth=0, facecolor='green', alpha=0.08, zorder=0)
    ax.add_patch(good_rect)

    plt.tight_layout()
    plt.savefig("caleb_blitz_stats.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = cleaned_data_blitz()
    create_scatter_blitz(df)