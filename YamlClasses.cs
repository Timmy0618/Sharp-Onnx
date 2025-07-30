using System.Collections.Generic;

namespace OnnxLibrary
{
    public class YamlClasses
    {
        public List<ClassInfo> Classes { get; set; }
    }

    public class ClassInfo
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
}
