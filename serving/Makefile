PROFILE ?= nvidia

INDEX_OUT := index.md
INDEX_NVIDIA_OUT := index_nvidia.md
INDEX_AMD_OUT := index_amd.md

INTRO_SNIPPET := snippets/intro_$(PROFILE).md
LEASE_SNIPPET := snippets/create_lease_$(PROFILE).md
SERVER_SNIPPET := snippets/create_server_$(PROFILE).md
EP_SNIPPET := snippets/ep_onnx_$(PROFILE).md

.PHONY: all build clean validate-profile

all: build

build: validate-profile $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) $(INDEX_OUT) 0_intro.ipynb 1_create_lease.ipynb 2_create_server.ipynb 3_prepare_data.ipynb 4_launch_jupyter.ipynb workspace/5_measure_torch.ipynb workspace/6_measure_onnx.ipynb workspace/7_optimize_onnx.ipynb workspace/8_ep_onnx.ipynb

validate-profile:
	@if [ "$(PROFILE)" != "nvidia" ] && [ "$(PROFILE)" != "amd" ]; then \
		echo "Unsupported PROFILE '$(PROFILE)'. Use PROFILE=nvidia or PROFILE=amd."; \
		exit 1; \
	fi

clean:
	rm -f $(INDEX_OUT) $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) \
		0_intro.ipynb \
		1_create_lease.ipynb \
		2_create_server.ipynb \
		3_prepare_data.ipynb \
		4_launch_jupyter.ipynb \
		workspace/5_measure_torch.ipynb \
		workspace/6_measure_onnx.ipynb \
		workspace/7_optimize_onnx.ipynb \
		workspace/8_ep_onnx.ipynb

$(INDEX_NVIDIA_OUT): snippets/intro_nvidia.md snippets/create_lease_nvidia.md snippets/create_server_nvidia.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx_nvidia.md snippets/footer.md
	cat snippets/intro_nvidia.md \
		snippets/create_lease_nvidia.md \
		snippets/create_server_nvidia.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx_nvidia.md \
		> index.nvidia.tmp.md
	grep -v '^:::' index.nvidia.tmp.md > $(INDEX_NVIDIA_OUT)
	rm index.nvidia.tmp.md
	cat snippets/footer.md >> $(INDEX_NVIDIA_OUT)

$(INDEX_AMD_OUT): snippets/intro_amd.md snippets/create_lease_amd.md snippets/create_server_amd.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx_amd.md snippets/footer.md
	cat snippets/intro_amd.md \
		snippets/create_lease_amd.md \
		snippets/create_server_amd.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx_amd.md \
		> index.amd.tmp.md
	grep -v '^:::' index.amd.tmp.md > $(INDEX_AMD_OUT)
	rm index.amd.tmp.md
	cat snippets/footer.md >> $(INDEX_AMD_OUT)

$(INDEX_OUT): index_$(PROFILE).md
	cp index_$(PROFILE).md $(INDEX_OUT)

0_intro.ipynb: $(INTRO_SNIPPET) validate-profile
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md $(INTRO_SNIPPET) \
				-o 0_intro.ipynb
	sed -i 's/attachment://g' 0_intro.ipynb

1_create_lease.ipynb: $(LEASE_SNIPPET) validate-profile
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md $(LEASE_SNIPPET) \
				-o 1_create_lease.ipynb
	sed -i 's/attachment://g' 1_create_lease.ipynb

2_create_server.ipynb: $(SERVER_SNIPPET) validate-profile
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md $(SERVER_SNIPPET) \
				-o 2_create_server.ipynb
	sed -i 's/attachment://g' 2_create_server.ipynb

3_prepare_data.ipynb: snippets/prepare_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/prepare_data.md \
				-o 3_prepare_data.ipynb
	sed -i 's/attachment://g' 3_prepare_data.ipynb

4_launch_jupyter.ipynb: snippets/launch_jupyter.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/launch_jupyter.md \
				-o 4_launch_jupyter.ipynb
	sed -i 's/attachment://g' 4_launch_jupyter.ipynb

workspace/5_measure_torch.ipynb: snippets/measure_torch.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/measure_torch.md \
				-o workspace/5_measure_torch.ipynb
	sed -i 's/attachment://g' workspace/5_measure_torch.ipynb

workspace/6_measure_onnx.ipynb: snippets/measure_onnx.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/measure_onnx.md \
				-o workspace/6_measure_onnx.ipynb
	sed -i 's/attachment://g' workspace/6_measure_onnx.ipynb

workspace/7_optimize_onnx.ipynb: snippets/optimize_onnx.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/optimize_onnx.md \
				-o workspace/7_optimize_onnx.ipynb
	sed -i 's/attachment://g' workspace/7_optimize_onnx.ipynb

workspace/8_ep_onnx.ipynb: $(EP_SNIPPET) validate-profile
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md $(EP_SNIPPET) \
				-o workspace/8_ep_onnx.ipynb
	sed -i 's/attachment://g' workspace/8_ep_onnx.ipynb
