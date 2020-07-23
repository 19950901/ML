二叉排序树删除节点的三种情况：

一、删除叶子节点
1、首先要先找到要删除的目标节点targetNode
2、找到targetNode的父节点parent
3、确定targetNode是parent的左子节点还是右子节点
4、若是左子节点则parent.left = None
   若是右子节点则parent.right = None

二、删除只有一棵子树的节点
1、首先要先找到要删除的目标节点targetNode
2、找到targetNode的父节点parent
3、确定targetNode的子节点是左子节点还是右子节点
4、确定targetNode是parent的左子节点还是右子节点
5、若targetNode有左子节点
    (1)若targetNode是parent的左子节点，parent.left=targetNode.left
    (2)若targetNode是parent的右子节点，parent.right=targetNode.left
6、若targetNode有右子节点
    (1)若targetNode是parent的左子节点，parent.left=targetNode.right
    (2)若targetNode是parent的右子节点，parent.right=targetNode.right

三、删除有两棵子树的节点
1、首先要先找到要删除的目标节点targetNode
2、找到targetNode的父节点parent
3、从targetNode的右子树找到值最小的节点
4、用临时变量temp将最小节点进行保存
5、删除该最小节点
6、targetNode.value = temp
或者
3、从targetNode的左子树找到值最大的节点
4、用临时变量temp将值最大节点进行保存
5、删除该值最大节点
6、targetNode.value = temp
