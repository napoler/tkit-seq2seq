## 快速上传操作
可以自动查找依赖，然后上传
```
sh upload.sh
```





更多开发说明参考这里 https://python-packaging-zh.readthedocs.io/zh_CN/latest/minimal.html


## 如何向PyPi(pip)提交模块

https://www.notion.so/terrychanorg/PyPi-pip-b371898f30ec4f268688edebab8d7ba1

## 提交到anaconda

https://docs.anaconda.com/anacondaorg/user-guide/tasks/work-with-packages/


##  MANIFEST.in 文件

 MANIFEST.in 文件，文件内容就是需要包含在分发包中的文件。一个 MANIFEST.in 文件如下：

```
include *.txt
recursive-include examples *.txt *.py
prune examples/sample?/build
```

MANIFEST.in 文件的编写规则可参考：https://docs.python.org/3.6/distutils/sourcedist.html
