Small lenguje models harness evalation with standard Benchmarks

```bash
#eval:
python main.py --model $(MODEL) --config configs/benchmarks.yaml

#eval-all:
python main.py --all --config configs/benchmarks.yaml

#report:\

python src/reporter.py --output reports/final_report.pdf

#dashboard:
streamlit run src/reporter.py

#docker-build:
docker compose -f docker/docker-compose.yml build

#docker-run:
docker compose -f docker/docker-compose.yml up
```

# make 