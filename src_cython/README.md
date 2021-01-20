## Prepare
1. [yukarin_autoreg_cpp](https://github.com/Hiroshiba/yukarin_autoreg_cpp)をビルド
2. 生成された`yukarin_autoreg_cpp.so`をこのディレクトリの`libyukarin_autoreg_cpp.so`にコピー（ファイル名に注意）

## Build
```bash
CFLAGS="-I." LDFLAGS="-L." python setup.py install
rm yukarin_autoreg_cpp.cpp; rm -r build

# リビルド時は以下も実行すると変なことになりにくい
python setup.py clean
```

## Check
```bash
# 実行結果が一緒かチェックするけど、randomサンプリングなので値が違う
PYTHONPATH="../" LD_LIBRARY_PATH="$LD_LIBRARY_PATH:." python check.py
```
