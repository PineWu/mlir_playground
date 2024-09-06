#!/bin/zsh

CURRENT_DIR=$(pwd)
LLVM_CC_JSON="/home/lixiang/Workspace/llvm-project/build/compile_commands.json"
SCH_CC_JSON="${CURRENT_DIR}/build/compile_commands.json"
TEMP_CC_JSON="/tmp/compile_commands.json"
FINAL_CC_JSON="${CURRENT_DIR}/compile_commands.json"


cp ${LLVM_CC_JSON} ${FINAL_CC_JSON}

# delete the last line of compile_commands.json
sed -i "$ d" "${FINAL_CC_JSON}"
# add a "," to end
echo "," >> ${FINAL_CC_JSON}


cp "${SCH_CC_JSON}" "${TEMP_CC_JSON}"

# delete the first line of scc compile_commands.json
sed -i "1 d" "${TEMP_CC_JSON}"

# concat two compile_commands.json files
cat "${TEMP_CC_JSON}" >> "${FINAL_CC_JSON}"
