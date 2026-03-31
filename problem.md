# 测试时发现的问题
1.有的 CVE task.yaml中的vulnerable_ref、fixed_ref和repo_url都跑不出来，例如CVE-2021-36976

2.二进制文件形式的poc模型无法识别，比如CVE-2022-28044，其poc文件应该是二进制文件，但是真正运行这个kbagent的时候无法将该poc跑出来。
