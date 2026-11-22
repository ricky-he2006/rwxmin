PLATFORM ?= linux/amd64
IMAGE ?= blast-radius-entry
SAMPLE := $(CURDIR)/sample
OUTPUT := $(CURDIR)/output.jsonl

# The flags the scorer runs your image with. The scorer is linux/amd64, so on
# an ARM machine this default requires emulation. Use PLATFORM=linux/arm64 for native timing.
RUN_FLAGS := --rm --platform $(PLATFORM) --network none --cpus 2 \
	--memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
	--tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
	--security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
	--volume $(SAMPLE)/commits:/data/commits:ro \
	--volume $(SAMPLE)/tests.json:/data/tests.json:ro

.PHONY: build sample check

build:
	docker build --platform $(PLATFORM) --tag $(IMAGE) .

# Score three fresh containers, including CPU consumed by selector workers.
sample: build
	python3 tools/benchmark.py "$(SAMPLE)" "$(OUTPUT)" $(RUN_FLAGS) $(IMAGE)

# The output's rules only: every line well formed and every commit answered.
# A wrong selection still passes here; make sample counts the misses.
check: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	python3 score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT) > /dev/null && echo "format ok"
