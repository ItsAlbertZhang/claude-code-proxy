import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const rootChangelog = fileURLToPath(new URL("../../CHANGELOG.md", import.meta.url));
const docsChangelog = fileURLToPath(
  new URL("../src/content/docs/reference/changelog.md", import.meta.url),
);

const content = await readFile(rootChangelog, "utf8");
const hasFrontmatter = content.startsWith("---\n") || content.startsWith("---\r\n");
if (!hasFrontmatter || !/\r?\ntitle: Changelog\r?\n/.test(content)) {
  throw new Error("CHANGELOG.md must contain Changelog frontmatter");
}
await writeFile(docsChangelog, content, "utf8");
