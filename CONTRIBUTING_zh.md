# R1-A7 工作区贡献说明

## 开始前

1. 从 `PROJECT_INDEX.md` 确认项目归属。
2. 从 `BRANCH_CATALOG.md` 确认当前统一工作线和历史快照关系。
3. 阅读项目 README 的真机安全边界。

## 分支和提交

- 日常功能分支使用 `codex/<project>-<topic>`。
- 单次实验可使用 `exp/<project>-<date>-<topic>`。
- 不从冻结快照继续长期开发；将必要变更带回当前统一工作线。
- 不强推或改写共享历史，不直接合并无共同祖先的 `main` 与后期工作区。

## 提交内容

- 提交源码、配置、可搜索文档和小型复核结果。
- 不提交凭据、个人信息、运行日志、原始大数据、视频、bag、模型权重或 checkpoint。
- 大文件使用清单、Git LFS 或外部存储，并记录 SHA256 和恢复方式。
- 新的真机控制入口必须写清通信、唯一控制源、限速、停止和急停方式。

## 本地检查

```bash
bash scripts/repository_sanity_check.sh
git diff --check
git status --short
```

通过仓库检查只代表结构和语法基本有效，不代表已完成真机安全验证。
