class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ""
        #"\t"*level+repr(self.value)+"\n"
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level+1)
            else:
                ret += "\t"*level + str(child) + "\n"
        return ret

    def add_child(self,c):
        if isinstance(c, type(self)):
            c.parent = self
        #self.children[self.num_children] = c
        self.children.append(c)
        self.num_children = self.num_children + 1

    def size(self):
        if self._size is not None:
            return self._size
        size = 1
        for i in range(self.num_children):
            size = size + self.children[i].size()
        self._size = size
        return size

    def children_vector(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                # non terminal
                r_list.append(4)
            else:
                r_list.append(self.children[i])
        return r_list

    def to_string(self):
        r_list = []
        #print("number of children: {}\n".format(self.num_children))
        #print("last child: {}\n".format(self.children[self.num_children-1]))
        for i in range(self.num_children):
            if isinstance(self.children[i],Tree):
                #print("is an instance of a tree")
                r_list.append("( " + self.children[i].to_string() + " )")
            else:
                #print("not a tree")
                #print(self.children[i])
                #print(str(self.children[i]))
                r_list.append(str(self.children[i]))
        return " ".join(r_list)

    def to_list(self, form_manager):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                r_list.append(form_manager.get_symbol_idx("("))
                cl = self.children[i].to_list(form_manager)
                for k in range(len(cl)):
                    r_list.append(cl[k])
                r_list.append(form_manager.get_symbol_idx(")"))
            else:
                r_list.append(self.children[i])
        return r_list
