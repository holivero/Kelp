import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import matplotlib.ticker as ticker

def run_simulations(SCENARIOS_DIR,EXTRA_PARAMS):
    for filename in os.listdir(SCENARIOS_DIR):
        if filename.endswith(".xlsx"):
            input_path = os.path.join(SCENARIOS_DIR, filename)
            print(f"Running simulation for: {filename}")
            subprocess.run(
                ["python3", "kelp_12092025.py", input_path] + EXTRA_PARAMS
            )



# 💡 Crea una función que formatea la etiqueta.
def scientific_formatter(x, pos):
    """
    Formatea la etiqueta para usar notación científica con un número fijo de decimales.
    """
    NUM_DECIMALS = 0

    if x == 0:
        return '0'
    else:
        # Aquí se determina el exponente en base a la magnitud del valor
        exponent = int(np.floor(np.log10(np.abs(x))))
        # El coeficiente se calcula para el número de decimales deseado
        mantissa = x / 10**exponent
        # Se devuelve la cadena formateada
        return f'{mantissa:.{NUM_DECIMALS}f}e{exponent}'

def load_results_and_plot(SCENARIOS_DIR, Parameter_to_compare):
    custom_colors = "colorblind"
    sns.set_palette(custom_colors)
    consolidated_data = []

    # === Cargar datos ===
    for full_name in os.listdir(SCENARIOS_DIR):
        name = full_name.split("-")[0]
        scenario_dir = os.path.join(SCENARIOS_DIR, full_name)
        if os.path.isdir(scenario_dir):
            result_file = os.path.join(scenario_dir, "data.npz")
            if os.path.exists(result_file):
                data = np.load(result_file)
                Parameter_to_compare_value = data[Parameter_to_compare]
                X = data['X']
                E = data['E']
                df = pd.DataFrame(X, columns=['Adult', 'Juvenile'])
                df['E'] = E
                df['Total Population'] = df['Juvenile'] + df['Adult']
                df['scenario'] = name
                df[Parameter_to_compare] = Parameter_to_compare_value
                consolidated_data.append(df)

    if not consolidated_data:
        print("No result files found.")
        return None

    df2 = pd.concat(consolidated_data, ignore_index=True)

    # ========= 1. Adult vs Juvenile =========
    # --- Conjunto + marginales ---
    custom_formatter = ticker.FuncFormatter(scientific_formatter)
    g = sns.JointGrid(data=df2, x="Adult", y="Juvenile", hue=Parameter_to_compare, height=8, ratio=4)
    g.plot_joint(sns.kdeplot, levels=5, cut=0, fill=False)
    g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.4, cut=0)
    sns.move_legend(g.ax_joint, "lower right")
    g.ax_joint.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'JointGrid_Adult_vs_Juvenile.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # --- Densidad conjunta sola ---
    plt.figure(figsize=(6, 6))
    sns.kdeplot(data=df2, x="Adult", y="Juvenile", hue=Parameter_to_compare, levels=5, fill=False, cut =0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'JointOnly_Adult_vs_Juvenile.png'), dpi=400)
    plt.close()

    # --- Marginal Adult ---
    plt.figure(figsize=(6, 2))
    ax =sns.kdeplot(data=df2, x="Adult", hue=Parameter_to_compare, fill=True, alpha=0.4, cut=0)
    ax.yaxis.set_major_formatter(custom_formatter)
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'Marginal_Adult.png'), dpi=400)
    plt.close()

    # --- Marginal Juvenile ---
    plt.figure(figsize=(6, 2))
    ax =sns.kdeplot(data=df2, x="Juvenile", hue=Parameter_to_compare, fill=True, alpha=0.4,cut=0)
    ax.yaxis.set_major_formatter(custom_formatter)
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'Marginal_Juvenile.png'), dpi=400)
    plt.close()

    # ========= 2. Total Population vs E =========
    # --- Conjunto + marginales ---
    g = sns.JointGrid(data=df2, x="Total Population", y="E", hue=Parameter_to_compare, height=8, ratio=4)
    g.plot_joint(sns.kdeplot, levels=5, cut=0, fill=False)
    g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.4, cut=0)
    sns.move_legend(g.ax_joint, "lower right")
    g.ax_joint.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'JointGrid_TotalPopulation_vs_E.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # --- Densidad conjunta sola ---
    plt.figure(figsize=(6, 6))
    sns.kdeplot(data=df2, x="Total Population", y="E", hue=Parameter_to_compare, levels=5, fill=False, cut=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'JointOnly_TotalPopulation_vs_E.png'), dpi=400)
    plt.close()

    # --- Marginal Total Population ---
    plt.figure(figsize=(6, 2))
    sns.kdeplot(data=df2, x="Total Population", hue=Parameter_to_compare, fill=True, alpha=0.4, cut=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'Marginal_TotalPopulation.png'), dpi=400)
    plt.close()

    # --- Marginal E ---
    plt.figure(figsize=(6, 2))
    sns.kdeplot(data=df2, x="E", hue=Parameter_to_compare, fill=True, alpha=0.4, cut=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIOS_DIR, f'Marginal_E.png'), dpi=400)
    plt.close()

    

    print("✅ Gráficos combinados y separados guardados en", SCENARIOS_DIR)
    return df2

'''
def load_results_and_plot(SCENARIOS_DIR,Parameter_to_compare):
    custom_colors = "colorblind"#["blue", "red", "black"]
    sns.color_palette(custom_colors)
    consolidated_data = []
    #print(os.listdir(SCENARIOS_DIR))
    for full_name in os.listdir(SCENARIOS_DIR):
        name = full_name.split("-")[0]
        scenario_dir = os.path.join(SCENARIOS_DIR, full_name)
        if os.path.isdir(scenario_dir):
            #print(scenario_dir)
            result_file = os.path.join(scenario_dir, "data.npz")
            #print(result_file)
            if os.path.exists(result_file):
                data = np.load(result_file)
                Parameter_to_compare_value = data[Parameter_to_compare]
                #print(Parameter_to_compare)
                #print(Parameter_to_compare_value)
                X = data['X']
                E = data['E']
                df = pd.DataFrame(X, columns=['Adult', 'Juvenile'])
                #print(df.head())
                df['E'] = E
                df['Total Population'] = df['Juvenile']+df['Adult']
                df["scenario"] = name
                df[Parameter_to_compare] = Parameter_to_compare_value
                consolidated_data.append(df)

    if consolidated_data:
        df2=pd.concat(consolidated_data, ignore_index=True)
    else:
        print("No result files found.")
        return None
    


    
    # Create the joint plot layout
    g = sns.JointGrid(data=df2, x="Adult", y="Juvenile", hue=Parameter_to_compare, height=8, ratio=4)
    
    # Plot KDE contours in main area
    g.plot_joint(sns.kdeplot, levels=5, cut=0, fill=False)

    # Plot marginal histograms
    g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.4,cut=0)

    sns.move_legend(g.ax_joint, "lower right")

    g.ax_joint.grid(True)
 
    # Ajustar layout para que nada se corte
    plt.tight_layout()


    # Save the figure
    fig_title = os.path.join(SCENARIOS_DIR, f'LevelSetHistogram_Adults_vs_Juveniles_different_{Parameter_to_compare}.png')
    plt.savefig(fig_title, dpi=400, bbox_inches='tight')
    plt.close()

    g = sns.JointGrid(data=df2, x="Total Population", y="E", hue=Parameter_to_compare, height=8, ratio=4)

    # Curvas de nivel KDE en el panel principal
    g.plot_joint(sns.kdeplot, levels=5, cut=0, fill=False)

    # Histogramas marginales suavizados con KDE
    g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.4, cut=0)

    # Mover la leyenda y agregar rejilla
    sns.move_legend(g.ax_joint, "lower right")
    g.ax_joint.grid(True)


    plt.tight_layout()  
    fig_title = os.path.join(SCENARIOS_DIR, f'LevelSetHistogram_Total_Population_vs_E_different_{Parameter_to_compare}.png')
    plt.savefig(fig_title, dpi=400, bbox_inches='tight')
    plt.close()

    return

'''
def main():
    try:
        parametersDir = sys.argv[1]
        yearsNoHarvesting = (sys.argv[2])
        yearsHarvesting = (sys.argv[3])
        numberOfTrajectories = (sys.argv[4])
        seed = (sys.argv[5])
        Parameter_to_compare = (sys.argv[6])
    except Exception as err:
        print('Please excecute as: python3 compare_scenarios.py parameters_dir yearsNoHarvesting yearsHarvesting numberOfTrajectories Random_Seed Parameter_to_compare')
        return
    
    SCENARIOS_DIR = parametersDir # Both input and output happen in this directory

    # Five global parameters for all simulations
    EXTRA_PARAMS = [yearsNoHarvesting, yearsHarvesting, numberOfTrajectories, seed, "1"]  

    run_simulations(SCENARIOS_DIR,EXTRA_PARAMS)
    load_results_and_plot(SCENARIOS_DIR,Parameter_to_compare)

if __name__ == "__main__":
    main()
