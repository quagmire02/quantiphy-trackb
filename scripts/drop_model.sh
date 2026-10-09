# Delete one Hugging Face model from the cache, including its files in the shared blob store
# (only blobs no other model links to). usage: drop_model.sh models--org--name
set -u
H=~/.cache/huggingface/hub
M="$1"
[ -d "$H/$M" ] || { echo "no $M"; exit 1; }
mine=$(mktemp); others=$(mktemp)
find "$H/$M" -type l -exec readlink -f {} \; | sort -u > "$mine"
find "$H" -path "$H/$M" -prune -o -path "$H/blobs" -prune -o -type l -print | xargs -r -n1 readlink -f | sort -u > "$others"
before=$(du -sh "$H" | cut -f1)
comm -23 "$mine" "$others" | while read -r f; do rm -f "$f" "$f.refs" "$f.lock"; done
rm -rf "$H/$M" "$H/.locks/$M"
find "$H/blobs" -mindepth 1 -type d -empty -delete 2>/dev/null
echo "cache $before -> $(du -sh "$H" | cut -f1); broken links: $(find "$H" -xtype l | wc -l)"
rm -f "$mine" "$others"
