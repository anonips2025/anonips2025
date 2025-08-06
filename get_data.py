import os
import openml


def load_data(datasets):

    for name, did in datasets.items():

        if os.path.exists(os.path.join(os.getcwd(), 'dataset', name, name + '.csv')):
            print('Dataset {} already exists'.format(name))
            continue
        print(f"Fetching dataset '{name}' (ID: {did})...")
        dataset = openml.datasets.get_dataset(did)
        X, y, categorical_columns, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

        os.makedirs(os.path.join(os.getcwd(), 'dataset', name), exist_ok=True)

        whole_data = X.copy()
        whole_data['class'] = y

        whole_data.to_csv(os.path.join(os.getcwd(), 'dataset', name, name + '.csv'), index=False, header=True)

        with open(os.path.join(os.getcwd(), 'dataset', name, name + '.info'), 'w') as f:
            for i,column in enumerate(attribute_names):
                f.write(f"{i+1} {'discrete' if categorical_columns[i] else 'continuous'}\n")
            f.write(f"class discrete\nLABEL_POS -1")


if __name__ == '__main__':

    # binary classification from https://huggingface.co/datasets/inria-soda/tabular-benchmark
    datasets_openml = {
        'electricity': 44156,
        'eye_movements': 44157,
        'credit_card_default': 45036,
        'covertype': 44159,
        'albert': 45035,
        'road_safety': 44161,
        'compas': 45039
    }

    load_data(datasets_openml)
