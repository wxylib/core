//! A small element tree: enough to read every VRT family and write it back.
//!
//! The typed models (see [`crate::classic`]) are views over this tree, and
//! keep any element or attribute they do not interpret, so a document read
//! and written again loses nothing but formatting.

use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};

use crate::Error;

/// One XML element: name, attributes in document order, children.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Element {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

/// A child of an element.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Element(Element),
    Text(String),
}

impl Element {
    pub fn new(name: &str) -> Self {
        Element {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Attribute value by name (exact case).
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Attribute value, trying each spelling in turn (VRT has a few
    /// historical case variants, such as `relativetoVRT`).
    pub fn attr_any(&self, keys: &[&str]) -> Option<&str> {
        keys.iter().find_map(|k| self.attr(k))
    }

    pub fn set_attr(&mut self, key: &str, value: impl Into<String>) {
        let value = value.into();
        match self.attrs.iter_mut().find(|(k, _)| k == key) {
            Some(kv) => kv.1 = value,
            None => self.attrs.push((key.to_string(), value)),
        }
    }

    /// Child elements, in order.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }

    /// First child element with this name.
    pub fn child(&self, name: &str) -> Option<&Element> {
        self.elements().find(|e| e.name == name)
    }

    /// Concatenated text content, trimmed.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for n in &self.children {
            if let Node::Text(t) = n {
                s.push_str(t);
            }
        }
        s.trim().to_string()
    }

    /// Text of the first child with this name.
    pub fn child_text(&self, name: &str) -> Option<String> {
        self.child(name).map(|e| e.text())
    }

    /// An element holding only text.
    pub fn with_text(name: &str, text: impl Into<String>) -> Self {
        Element {
            name: name.to_string(),
            attrs: vec![],
            children: vec![Node::Text(text.into())],
        }
    }

    pub fn push(&mut self, e: Element) {
        self.children.push(Node::Element(e));
    }
}

/// Parse a document and return its root element. Comments, processing
/// instructions and the XML declaration are dropped; whitespace-only text
/// between elements is dropped.
pub fn parse(text: &str) -> Result<Element, Error> {
    let mut reader = Reader::from_str(text);
    let mut stack: Vec<Element> = vec![Element::new("#root")];
    loop {
        match reader.read_event().map_err(|e| Error::Xml(e.to_string()))? {
            Event::Start(s) => stack.push(start(&s)?),
            Event::Empty(s) => {
                let e = start(&s)?;
                stack.last_mut().unwrap().push(e);
            }
            Event::End(_) => {
                let e = stack.pop().unwrap();
                if stack.is_empty() {
                    return Err(Error::Xml("unbalanced end tag".into()));
                }
                stack.last_mut().unwrap().push(e);
            }
            Event::Text(t) => {
                let s = t.decode().map_err(|e| Error::Xml(e.to_string()))?;
                let s = quick_xml::escape::unescape(&s).map_err(|e| Error::Xml(e.to_string()))?;
                // Text around an entity reference arrives in pieces: join
                // them, keeping whitespace between pieces (`&quot; &quot;`).
                let top = stack.last_mut().unwrap();
                match top.children.last_mut() {
                    Some(Node::Text(t)) => t.push_str(&s),
                    _ if !s.trim().is_empty() => top.children.push(Node::Text(s.into_owned())),
                    _ => {}
                }
            }
            Event::GeneralRef(r) => {
                let name = r.decode().map_err(|e| Error::Xml(e.to_string()))?;
                let s = match &*name {
                    "lt" => "<".to_string(),
                    "gt" => ">".to_string(),
                    "amp" => "&".to_string(),
                    "quot" => "\"".to_string(),
                    "apos" => "'".to_string(),
                    n if n.starts_with("#x") => u32::from_str_radix(&n[2..], 16)
                        .ok()
                        .and_then(char::from_u32)
                        .map(String::from)
                        .ok_or_else(|| Error::Xml(format!("bad reference &{n};")))?,
                    n if n.starts_with('#') => n[1..]
                        .parse::<u32>()
                        .ok()
                        .and_then(char::from_u32)
                        .map(String::from)
                        .ok_or_else(|| Error::Xml(format!("bad reference &{n};")))?,
                    n => return Err(Error::Xml(format!("unknown entity &{n};"))),
                };
                let top = stack.last_mut().unwrap();
                match top.children.last_mut() {
                    Some(Node::Text(t)) => t.push_str(&s),
                    _ => top.children.push(Node::Text(s)),
                }
            }
            Event::CData(t) => {
                let s = String::from_utf8_lossy(&t).into_owned();
                stack.last_mut().unwrap().children.push(Node::Text(s));
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if stack.len() != 1 {
        return Err(Error::Xml("unclosed element".into()));
    }
    let root = stack.pop().unwrap();
    let first = root.elements().next().cloned();
    first.ok_or_else(|| Error::Xml("no root element".into()))
}

fn start(s: &BytesStart) -> Result<Element, Error> {
    let name = String::from_utf8_lossy(s.name().as_ref()).into_owned();
    let mut attrs = Vec::new();
    for a in s.attributes() {
        let a = a.map_err(|e| Error::Xml(e.to_string()))?;
        let k = String::from_utf8_lossy(a.key.as_ref()).into_owned();
        let v = a
            .unescape_value()
            .map_err(|e| Error::Xml(e.to_string()))?
            .into_owned();
        attrs.push((k, v));
    }
    Ok(Element {
        name,
        attrs,
        children: vec![],
    })
}

/// Write an element as an indented document (two spaces).
pub fn write(root: &Element) -> String {
    let mut w = Writer::new_with_indent(Vec::new(), b' ', 2);
    write_el(&mut w, root);
    let mut s = String::from_utf8(w.into_inner()).expect("utf8");
    s.push('\n');
    s
}

fn write_el(w: &mut Writer<Vec<u8>>, e: &Element) {
    let mut s = BytesStart::new(e.name.as_str());
    for (k, v) in &e.attrs {
        s.push_attribute((k.as_str(), v.as_str()));
    }
    if e.children.is_empty() {
        w.write_event(Event::Empty(s)).unwrap();
        return;
    }
    w.write_event(Event::Start(s)).unwrap();
    for n in &e.children {
        match n {
            Node::Element(c) => write_el(w, c),
            Node::Text(t) => w.write_event(Event::Text(BytesText::new(t))).unwrap(),
        }
    }
    w.write_event(Event::End(BytesEnd::new(e.name.as_str())))
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_attributes_text_and_entities() {
        let src = r#"<?xml version="1.0"?>
<!-- comment -->
<A x="1" y="a&amp;b"><B>t &lt; u</B><C/><D>1, 2</D></A>"#;
        let a = parse(src).unwrap();
        assert_eq!(a.attr("y"), Some("a&b"));
        assert_eq!(a.child_text("B").as_deref(), Some("t < u"));
        let again = parse(&write(&a)).unwrap();
        assert_eq!(a, again);
    }

    /// Quotes written raw and written as entities read the same, so a
    /// document with WKT in <SRS> survives a write and read.
    #[test]
    fn text_with_entities_is_one_node() {
        let raw = parse(r#"<SRS>PROJCS["WGS 84", AXIS["E" ,EAST]]</SRS>"#).unwrap();
        let esc =
            parse("<SRS>PROJCS[&quot;WGS 84&quot;, AXIS[&quot;E&quot; ,EAST]]</SRS>").unwrap();
        assert_eq!(raw, esc);
        assert_eq!(raw.children.len(), 1);
        assert_eq!(parse(&write(&raw)).unwrap(), raw);
        let sp = parse("<T>&quot; &quot;</T>").unwrap();
        assert_eq!(sp.text(), "\" \"");
    }
}
