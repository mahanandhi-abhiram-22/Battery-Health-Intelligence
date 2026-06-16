# generate_datasets.py
from train_model import BatteryTrainingPipeline
import pandas as pd
import numpy as np
import os

def generate_synth_csvs(outdir='data'):
    os.makedirs(outdir, exist_ok=True)
    # create three synthetic series with varying lengths
    def make_series(n_cycles, base_v=3.7, base_i=1.0, base_temp=25.0, seed=0):
        np.random.seed(seed)
        cycles = np.arange(1, n_cycles+1)
        alpha = np.random.uniform(1.2,1.8)
        soh = 1.0 - (cycles / (n_cycles*0.9))**alpha
        soh = np.clip(soh + np.random.normal(0, 0.01, size=len(soh)), 0.1, 1.0)
        voltage = base_v - (1-soh)*0.25 + np.random.normal(0,0.01,size=len(soh))
        current = base_i + (1-soh)*0.4 + np.random.normal(0,0.05,size=len(soh))
        temperature = base_temp + (1-soh)*10.0 + np.random.normal(0,1.0,size=len(soh))
        rul = ((soh - 0.8)/ (0.0001 + 1e-9)) / 365.0
        df = pd.DataFrame({'voltage':voltage,'current':current,'temperature':temperature,'cycle':cycles,'soh':soh,'rul':rul})
        return df

    n1 = make_series(600, seed=7)
    n2 = make_series(800, base_v=3.75, base_i=0.9, seed=11)
    n3 = make_series(1000, base_v=3.68, base_i=1.2, seed=19)
    n1.to_csv(os.path.join(outdir,'nasa_battery_synth.csv'), index=False)
    n2.to_csv(os.path.join(outdir,'oxford_battery_synth.csv'), index=False)
    n3.to_csv(os.path.join(outdir,'calce_battery_synth.csv'), index=False)
    print("Saved synthetic CSVs to", outdir)

if __name__=='__main__':
    generate_synth_csvs()
