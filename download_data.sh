mkdir -p data

# Download the data
for dataset in "50-218" "100-430" "250-1065"; do
    wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf${dataset}.tar.gz -O data/uf${dataset}.tar.gz
    tar -xzf data/uf${dataset}.tar.gz -C data/
    wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf${dataset}.tar.gz -O data/uuf${dataset}.tar.gz
    tar -xzf data/uuf${dataset}.tar.gz -C data/
done

mkdir -p data/uf50-218
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz -O data/uf50-218.tar.gz
tar -xzf data/uf50-218.tar.gz -C data/uf50-218/

mkdir -p data/uuf50-218
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf50-218.tar.gz -O data/uuf50-218.tar.gz
tar -xzf data/uuf50-218.tar.gz --strip-components=1 -C data/uuf50-218/

mkdir -p data/uf100-430
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz -O data/uf100-430.tar.gz
tar -xzf data/uf100-430.tar.gz -C data/uf100-430/

mkdir -p data/uuf100-430
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf100-430.tar.gz -O data/uuf100-430.tar.gz
tar -xzf data/uuf100-430.tar.gz --strip-components=1 -C data/uuf100-430/

mkdir -p data/uf250-1065
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz -O data/uf250-1065.tar.gz
tar -xzf data/uf250-1065.tar.gz --strip-components=4 -C data/uf250-1065/

mkdir -p data/uuf250-1065
wget https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf250-1065.tar.gz -O data/uuf250-1065.tar.gz
tar -xzf data/uuf250-1065.tar.gz --strip-components=1 -C data/uuf250-1065/