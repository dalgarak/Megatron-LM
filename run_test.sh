#./train_wbl_test1_250807.sh ./wbl-small-test/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=1 ./train_wbl_test2_250808.sh ./wbl-depth-test/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=1 ./train_wbl_test3_250808_fp8.sh ./wbl-depth-test-fp8/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_FLASH_ATTN=1 ./train_wbl_100b_test_250809_fp8.sh ./wbl-100b-half_layer-moe-test-fp8/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_FLASH_ATTN=1 ./calc_test_wbl_100b_test_250809_fp8.sh ./wbl-100b-half_layer-moe-test-fp8/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document

#NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 NVTE_FLASH_ATTN=1 ./train_wbl_test2_dense_no_sliding_window.sh ./wbl-d2048-l24-attnh16-no-sliding/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 NVTE_FLASH_ATTN=1 ./train_wbl_100b_dense_as_one_expert.sh ./wbl-dense-48l-3072d-mlp_intrm16384_swa5n1-mla/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 NVTE_FLASH_ATTN=1 ./train_wbl_100b_dense_2_expert_1shared_moe_upcycle.sh ./wbl-dense-48l-3072d-mlp_intrm1536_swa5n1-mla/ ./wbl-moe-2exp-1shared-48l-3072d-mlp_intrm1536_swa5n1-mla/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 NVTE_FLASH_ATTN=1 ./train_wbl_100b_test_250809_fp8.sh ./wbl-dense-48l-3072d-mlp_intrm1536_swa5n1-mla/ ./wbl-moe-2exp-1shared-48l-3072d-mlp_intrm1536_swa5n1-mla/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
