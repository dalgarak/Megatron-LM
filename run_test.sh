#./train_wbl_test1_250807.sh ./wbl-small-test/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=1 ./train_wbl_test2_250808.sh ./wbl-depth-test/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
#NVTE_DEBUG=0 NVTE_DEBUG_LEVEL=0 NVTE_FLASH_ATTN=1 ./train_wbl_test3_250808_fp8.sh ./wbl-depth-test-fp8/ ~/egpt-models/eagle-3b-preview/ ../hftb_finemath_4plus_egpt-3b-tokenizer-megatron_text_document
