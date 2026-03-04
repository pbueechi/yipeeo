import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dhf():
    path = r'M:\Projects\Clim4Cast\04_deliverables_documents\Activity_1.1\DHF_Database\DHF_events_and_impacts_database.csv'
    a = pd.read_csv(path, sep=";")
    years = a.Year.value_counts().sort_index()
    print(years)
    years.plot(kind='bar')
    # plt.bar(years)
    plt.show()

def reshape_crop_csv():
    path = r'M:\Projects\Clim4Cast\04_deliverables_documents\Activity_1.1\Crop_yields\Delivered'
    template_path = os.path.join(path, 'template_germany.csv')
    austria_path = os.path.join(path, 'crop_yield_austria_nuts3.csv')
    path_nuts = os.path.join(path, 'nuts3_names.csv')

    temp = pd.read_csv(template_path)
    austria = pd.read_csv(austria_path)
    nuts_dict = dict(pd.read_csv(path_nuts, sep=';').values)

    austria = austria.rename(columns={'Unit':'Region_name', 'Region_number':'Region_id'})
    austria.iloc[:,3:] = austria.iloc[:,3:]/10
    austria.iloc[:, 3:] = austria.iloc[:,3:].round(2)
    austria.Region_name = austria.Region_id
    austria.Region_name = austria.Region_name.replace(nuts_dict)
    crop_types = np.unique(austria.Crop_type)
    for crop_type in crop_types:
        inds = np.where(austria.Crop_type==crop_type)[0]
        this_df = austria.iloc[inds,:]
        this_df = this_df.drop('Crop_type', axis=1)
        this_df.to_csv(f'{path}/{crop_type}_Austria.csv', index=False)


if __name__ == '__main__':
    reshape_crop_csv()