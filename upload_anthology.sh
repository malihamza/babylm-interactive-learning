# Set your dataset repo
REPO=llm-slice/storytelling_anthology
BASE_DIR=./data_ppo_training/

for folder in "$BASE_DIR"/blm-gpt2s-90M*-seed42*; do
    [ -d "$folder/meta_data" ] || continue
    b=$(basename "$folder")
    branch=$(echo "$b" | grep -o 'chck_[0-9]\+M')
    if [ -z "$branch" ]; then
        echo "Could not parse branch for $b, skipping."
        continue
    fi
    echo "Uploading ALL files in $folder/meta_data to root of branch $branch in $REPO"
    hf upload $REPO \
        "$folder/meta_data" . \
        --repo-type dataset \
        --commit-message "Upload meta_data for $b" \
        --revision $branch
done