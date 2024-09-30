-- RUN: executor-runner %s -split-input-file | FileCheck %s

function main()
  print("segment 1")
  return 0
end

-- CHECK-LABEL: segment 1

-- // -----

function main()
  print("segment 2")
  return 0
end
-- CHECK-LABEL: segment 2
