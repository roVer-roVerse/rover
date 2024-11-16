import datasets as ds


def group_by(data: ds.Dataset, key: str) -> ds.DatasetDict:
    """
    Groups a dataset into a dictionary of datasets based on a specified key column.

    Args:
        data (ds.Dataset): The input dataset to be grouped.
        key (str): The column name in the dataset on which to group the data.

    Returns:
        ds.DatasetDict: A dictionary of datasets, where the keys are unique values
                        from the specified column, and the values are subsets of
                        the original dataset corresponding to each unique key.

    Example:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({"category": ["A", "B", "A"], "value": [1, 2, 3]})
        >>> grouped = group_by(data, "category")
        >>> print(grouped)
        DatasetDict({
            'A': Dataset({
                features: ['category', 'value'],
                num_rows: 2
            }),
            'B': Dataset({
                features: ['category', 'value'],
                num_rows: 1
            })
        })
    """
    it = data.select_columns([key]).to_pandas()[key]
    group_names = it.unique()
    xs = ds.DatasetDict()
    for g in group_names:
        mask = it[it == g]
        mask = mask.index.to_numpy()
        xs[g] = data.select(mask, keep_in_memory=True)
    return xs


def bucket_by(
    data: ds.Dataset, key: str, n: int, bucket_col_name="bucket_id", num_proc=None
) -> ds.DatasetDict:
    """
    Buckets a dataset into approximately `n` groups based on a hash of the values
    in the specified key column. Each bucket is assigned a unique ID.

    Args:
        data (ds.Dataset): The input dataset to be bucketed.
        key (str): The column name in the dataset to use for bucketing.
        n (int): The number of buckets to create.
        bucket_col_name (str, optional): The name of the column to store bucket IDs. Defaults to "bucket_id".
        num_proc (int, optional): The number of processes to use for parallelization. Defaults to None.

    Returns:
        ds.DatasetDict: A dictionary of datasets, where the keys are bucket IDs,
                        and the values are subsets of the original dataset corresponding
                        to each bucket.

    Example:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
        >>> bucketed = bucket_by(data, "id", n=2)
        >>> print(bucketed)
        DatasetDict({
            '0': Dataset({
                features: ['id', 'value', 'bucket_id'],
                num_rows: 2
            }),
            '1': Dataset({
                features: ['id', 'value', 'bucket_id'],
                num_rows: 2
            })
        })
    """
    data = data.map(
        lambda x: {bucket_col_name: str(hash(x) % n)},
        keep_in_memory=True,
        num_proc=num_proc,
        input_columns=[key],
    )
    return group_by(data, key=bucket_col_name)
