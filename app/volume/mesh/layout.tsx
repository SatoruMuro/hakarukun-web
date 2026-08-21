import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "3Dモデルから体積を測る｜体積ハカルくん",
  description: "Scaniverseなどで作成したGLBを端末内で切り出し、対象物の体積を推定するWebツール",
};

export default function MeshVolumeLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
